import os
import time
import torch
import numpy as np
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

from solnml.utils.constant import MAX_INT
from solnml.components.utils.constants import IMG_CLS, TEXT_CLS, OBJECT_DET
from solnml.components.metrics.metric import get_metric
from solnml.utils.logging_utils import setup_logger, get_logger
from solnml.components.ensemble.dl_ensemble.ensemble_bulider import ensemble_list
from solnml.components.evaluators.dl_evaluator import DLEvaluator
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space
from solnml.components.utils.mfse_utils.config_space_utils import sample_configurations
from solnml.components.computation.parallel_evaluator import ParallelEvaluator


class AutoDLBase(object):
    def __init__(self, time_limit=300,
                 trial_num=None,
                 dataset_name='default_name',
                 task_type=IMG_CLS,
                 metric='acc',
                 include_algorithms=None,
                 ensemble_method='ensemble_selection',
                 ensemble_size=50,
                 config_file_path=None,
                 evaluation='holdout',
                 logging_config=None,
                 output_dir="logs/",
                 random_state=1,
                 n_jobs=1):
        from solnml.components.models.img_classification import _classifiers as _img_estimators, _addons as _img_addons
        from solnml.components.models.text_classification import _classifiers as _text_estimators, \
            _addons as _text_addons
        from solnml.components.models.object_detection import _classifiers as _od_estimators, _addons as _od_addons

        self.metric_id = metric
        self.metric = get_metric(self.metric_id)

        self.dataset_name = dataset_name
        self.time_limit = time_limit
        self.trial_num = trial_num
        self.seed = random_state
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logging_config = logging_config
        self.logger = self._get_logger(self.dataset_name)

        self.evaluation_type = evaluation
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.n_jobs = n_jobs

        self.config_file_path = config_file_path
        self.update_cs = None

        if include_algorithms is not None:
            self.include_algorithms = include_algorithms
        else:
            if task_type == IMG_CLS:
                self.include_algorithms = list(_img_estimators.keys())
            elif task_type == TEXT_CLS:
                self.include_algorithms = list(_text_estimators.keys())
            else:
                raise ValueError("Unknown task type %s" % task_type)

        if task_type == IMG_CLS:
            self._estimators = _img_estimators
            self._addons = _img_addons
        elif task_type == TEXT_CLS:
            self._estimators = _text_estimators
            self._addons = _text_addons
        elif task_type == OBJECT_DET:
            self._estimators = _od_estimators
            self._addons = _od_addons
        else:
            raise ValueError("Unknown task type %s" % task_type)

        if ensemble_method is not None and ensemble_method not in ensemble_list:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)
        self.es = None
        self.solvers = dict()
        self.evaluators = dict()
        # Single model.
        self.best_algo_id = None
        self.best_algo_config = None
        # Ensemble models.
        self.candidate_algo_ids = None
        self.device = self.get_device()

        # Neural architecture selection.
        self.nas_evaluator, self.executor = None, None
        self.eval_hist_configs = dict()
        self.eval_hist_perfs = dict()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        return device

    def _get_logger(self, name):
        logger_name = 'SolnML-%s(%d)' % (name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)

    def get_model_config_space(self, estimator_id, include_estimator=True):
        if estimator_id in self._estimators:
            clf_class = self._estimators[estimator_id]
        elif estimator_id in self._addons.components:
            clf_class = self._addons.components[estimator_id]
        else:
            raise ValueError("Algorithm %s not supported!" % estimator_id)

        cs = clf_class.get_hyperparameter_search_space()
        model = UnParametrizedHyperparameter("estimator", estimator_id)
        if include_estimator:
            cs.add_hyperparameter(model)
        if self.task_type == IMG_CLS:
            aug_space = get_aug_hyperparameter_space()
            cs.add_hyperparameters(aug_space.get_hyperparameters())
            cs.add_conditions(aug_space.get_conditions())
        return cs

    def profile_models(self, profile_epoch_n=1):
        training_cost_per_unit = list()
        default_training_epoch = list()
        for estimator_id in self.include_algorithms:
            cs = self.get_model_config_space(estimator_id)
            default_config = cs.get_default_configuration()
            assert 'epoch_num' in default_config
            default_training_epoch.append(default_config['epoch_num'])
            cs.seed(self.seed)

            hpo_evaluator = self.evaluators[estimator_id]
            _start_time = time.time()
            try:
                hpo_evaluator(default_config, profile_epoch=profile_epoch_n)
                training_cost_per_unit.append(time.time() - _start_time)
            except Exception as e:
                print(e)
                training_cost_per_unit.append(MAX_INT)

        K = 5
        estimator_list = list()
        for id, estimator_id in enumerate(self.include_algorithms):
            num_of_units = default_training_epoch[id] / profile_epoch_n
            if training_cost_per_unit[id] * num_of_units * K < self.time_limit:
                estimator_list.append(estimator_id)
        return estimator_list

    def sample_configs_for_archs(self, include_architectures, N):
        configs = list()
        for _arch in include_architectures:
            _cs = self.get_model_config_space(_arch)
            configs.extend(sample_configurations(_cs, N))
        return configs

    def exec_SEE(self, architecture_candidates):
        eta, N, R = 3, 9, 27
        r = 1
        C = self.sample_configs_for_archs(architecture_candidates, N)

        """
            iteration procedure.
            iter_1: r=1, n=5*9
            iter_2: r=3, n=5*3
            iter_3: r=9, n=5
        """
        while r < R:
            if r not in self.eval_hist_configs:
                self.eval_hist_configs[r] = list()
            if r not in self.eval_hist_perfs:
                self.eval_hist_perfs[r] = list()

            val_losses = self.executor.parallel_execute(C, resource_ratio=float(r / R))
            for _id, _val_loss in enumerate(val_losses):
                if np.isfinite(_val_loss):
                    self.eval_hist_configs[r].append(C[_id])
                    self.eval_hist_perfs[r].append(_val_loss)

            # Select a number of best configurations for the next loop.
            # Filter out early stops, if any.
            indices = np.argsort(val_losses)
            if len(C) >= eta:
                C = [C[i] for i in indices]
                reduced_num = int(len(C) / eta)
                C = C[0:reduced_num]
            else:
                C = [C[indices[0]]]
            r *= eta
        return [config['estimator'] for config in C]

    def select_network_architectures(self, algorithm_candidates, train_data, num_arch=1):
        self.nas_evaluator = DLEvaluator(None,
                                         self.task_type,
                                         scorer=self.metric,
                                         dataset=train_data,
                                         device=self.device,
                                         seed=self.seed)
        self.executor = ParallelEvaluator(self.nas_evaluator, n_worker=self.n_jobs)

        _archs = algorithm_candidates.copy()
        while len(_archs) > num_arch:
            _archs = self.exec_SEE(_archs)
        return _archs
