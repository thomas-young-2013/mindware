import os
import time
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

from solnml.utils.constant import MAX_INT
from solnml.components.utils.constants import IMG_CLS, TEXT_CLS, OBJECT_DET
from solnml.components.metrics.metric import get_metric
from solnml.utils.logging_utils import setup_logger, get_logger
from solnml.components.ensemble.dl_ensemble.ensemble_bulider import ensemble_list
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space
from solnml.components.optimizers.base.config_space_utils import sample_configurations
from solnml.components.computation.parallel_process import ParallelProcessEvaluator

profile_image_size = [32, 128, 256]
profile_ratio = {
    'p100': {
        32: {
            'densenet121': 2.87,
            'densenet161': 3.24,
            'efficientnet': 1.79,
            'mobilenet': 1.07,
            'nasnet': 4.76,
            'resnet34': 1,
            'resnet50': 1.34,
            'resnet101': 2.07,
            'resnet152': 2.7,
            'resnext': 3.37,
            'senet': 4.03
        },
        128: {
            'densenet121': 2.57,
            'densenet161': 2.73,
            'efficientnet': 1.62,
            'mobilenet': 1,
            'nasnet': 4.39,
            'resnet34': 1.01,
            'resnet50': 1.56,
            'resnet101': 2.55,
            'resnet152': 3.56,
            'resnext': 4.13,
            'senet': 4.74
        },
        256: {
            'densenet121': 2.72,
            'densenet161': 2.68,
            'efficientnet': 1.76,
            'mobilenet': 1,
            'nasnet': 3.77,
            'resnet34': 1.18,
            'resnet50': 2.13,
            'resnet101': 3.34,
            'resnet152': 4.65,
            'resnext': 4.02,
            'senet': 4.48
        }
    },
    'titan rtx': {
        32: {
            'densenet121': 2.89,
            'densenet161': 3.3,
            'efficientnet': 1.66,
            'mobilenet': 1.08,
            'nasnet': 5.2,
            'resnet34': 1,
            'resnet50': 1.22,
            'resnet101': 1.9,
            'resnet152': 2.76,
            'resnext': 3.21,
            'senet': 3.73
        },
        128: {
            'densenet121': 2.68,
            'densenet161': 2.75,
            'efficientnet': 1.5,
            'mobilenet': 1,
            'nasnet': 4.53,
            'resnet34': 0.99,
            'resnet50': 1.54,
            'resnet101': 2.33,
            'resnet152': 3.22,
            'resnext': 2.83,
            'senet': 3.27
        },
        256: {
            'densenet121': 3.09,
            'densenet161': 2.87,
            'efficientnet': 1.83,
            'mobilenet': 1,
            'nasnet': 3.82,
            'resnet34': 1.37,
            'resnet50': 2.48,
            'resnet101': 3.89,
            'resnet152': 5.43,
            'resnext': 3.16,
            'senet': 3.58
        }
    }
}


class AutoDLBase(object):
    def __init__(self, time_limit=300,
                 trial_num=None,
                 dataset_name='default_name',
                 task_type=IMG_CLS,
                 metric='acc',
                 include_algorithms=None,
                 ensemble_method='ensemble_selection',
                 ensemble_size=50,
                 max_epoch=150,
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
        self.termination_time = time.time() + self.time_limit
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
        self.update_cs = dict()

        if include_algorithms is not None:
            self.include_algorithms = include_algorithms
        else:
            if task_type == IMG_CLS:
                self.include_algorithms = list(_img_estimators.keys())
            elif task_type == TEXT_CLS:
                self.include_algorithms = list(_text_estimators.keys())
            elif task_type == OBJECT_DET:
                self.include_algorithms = list(_od_estimators.keys())
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
        self.device = 'cuda'

        # Neural architecture selection.
        self.nas_evaluator = None
        self.eval_hist_configs = dict()
        self.eval_hist_perfs = dict()

        self.max_epoch = max_epoch
        self.image_size = None

    def _get_logger(self, name):
        logger_name = 'SolnML-%s(%d)' % (name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)

    def get_model_config_space(self, estimator_id, include_estimator=True, include_aug=True):
        if estimator_id in self._estimators:
            clf_class = self._estimators[estimator_id]
        elif estimator_id in self._addons.components:
            clf_class = self._addons.components[estimator_id]
        else:
            raise ValueError("Algorithm %s not supported!" % estimator_id)

        default_cs = clf_class.get_hyperparameter_search_space()
        model = UnParametrizedHyperparameter("estimator", estimator_id)
        if include_estimator:
            default_cs.add_hyperparameter(model)
        if self.task_type == IMG_CLS and include_aug is True:
            aug_space = get_aug_hyperparameter_space()
            default_cs.add_hyperparameters(aug_space.get_hyperparameters())
            default_cs.add_conditions(aug_space.get_conditions())

        # Update configuration space according to config file
        all_cs = self.update_cs.get('all', ConfigurationSpace())
        all_hp_names = all_cs.get_hyperparameter_names()
        estimator_cs = self.update_cs.get(estimator_id, ConfigurationSpace())
        estimator_hp_names = estimator_cs.get_hyperparameter_names()

        cs = ConfigurationSpace()
        for hp_name in default_cs.get_hyperparameter_names():
            if hp_name in estimator_hp_names:
                cs.add_hyperparameter(estimator_cs.get_hyperparameter(hp_name))
            elif hp_name in all_hp_names:
                cs.add_hyperparameter(all_cs.get_hyperparameter(hp_name))
            else:
                cs.add_hyperparameter(default_cs.get_hyperparameter(hp_name))

        cond = default_cs.get_conditions()
        cs.add_conditions(cond)

        return cs

    def profile_models(self, num_samples):
        profile_iter = 200
        training_costs = list()
        ref_time_cost = 0
        if self.task_type == IMG_CLS:
            from solnml.components.models.img_classification import _classifiers
            builtin_classifiers = _classifiers.keys()
        elif self.task_type == TEXT_CLS:
            from solnml.components.models.text_classification import _classifiers
            builtin_classifiers = _classifiers.keys()
        elif self.task_type == OBJECT_DET:
            from solnml.components.models.object_detection import _classifiers
            builtin_classifiers = _classifiers.keys()
        else:
            raise ValueError("Invalid task type %s" % self.task_type)
        for estimator_id in self.include_algorithms:
            if self.task_type == IMG_CLS:
                if estimator_id in builtin_classifiers:
                    # Get time cost for reference (use MobileNet as reference model)
                    if not ref_time_cost:
                        cs = self.get_model_config_space('mobilenet')
                        default_config = cs.get_default_configuration()
                        cs.seed(self.seed)

                        hpo_evaluator = self.evaluators[estimator_id]
                        try:
                            ref_time_cost = hpo_evaluator(default_config,
                                                          # profile_epoch=profile_epoch_n,
                                                          profile_iter=profile_iter,
                                                          )

                        except Exception as e:
                            self.logger.error(e)
                            self.logger.error('Reference model for profile failed! All base models will be chosen!')
                            ref_time_cost = 0

                    cs = self.get_model_config_space(estimator_id)
                    default_config = cs.get_default_configuration()
                    default_batch_size = default_config['batch_size']
                    # TODO: hardware device
                    device = 'p100'
                    nearest_image_size = None
                    distance = np.inf
                    for possible_image_size in profile_image_size:
                        if abs(self.image_size - possible_image_size) < distance:
                            nearest_image_size = possible_image_size
                            distance = abs(self.image_size - possible_image_size)
                    time_cost = ref_time_cost * profile_ratio[device][nearest_image_size][estimator_id] / \
                                profile_ratio[device][nearest_image_size]['mobilenet']
                    time_cost = time_cost * self.max_epoch * num_samples / default_batch_size / profile_iter
                else:
                    time_cost = 0

            else:
                cs = self.get_model_config_space(estimator_id)
                default_config = cs.get_default_configuration()
                default_batch_size = default_config['batch_size']
                cs.seed(self.seed)

                hpo_evaluator = self.evaluators[estimator_id]
                try:
                    time_cost = hpo_evaluator(default_config,
                                              # profile_epoch=profile_epoch_n,
                                              profile_iter=profile_iter,
                                              )
                    time_cost *= time_cost * self.max_epoch * (num_samples / default_batch_size) / profile_iter

                except Exception as e:
                    self.logger.error(e)
                    time_cost = MAX_INT

            training_costs.append(time_cost)

        K = 5
        estimator_list = list()
        for id, estimator_id in enumerate(self.include_algorithms):
            if training_costs[id] * K < self.time_limit:
                estimator_list.append(estimator_id)
        return estimator_list

    def sample_configs_for_archs(self, include_architectures, N):
        configs = list()
        for _arch in include_architectures:
            _cs = self.get_model_config_space(_arch)
            configs.extend(sample_configurations(_cs, N))
        return configs

    def exec_SEE(self, architecture_candidates, executor=None):
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
            _start_time = time.time()
            self.logger.info('Evaluate %d configurations with %d resource' % (len(C), r))

            if self.n_jobs > 1:
                val_losses = executor.parallel_execute(C, resource_ratio=float(r / R), eta=eta, first_iter=(r == 1))
            else:
                val_losses = list()
                for _config in C:
                    _result = self.nas_evaluator(_config, resource_ratio=float(r / R), eta=eta, first_iter=(r == 1))
                    val_losses.append(_result)

            for _id, _val_loss in enumerate(val_losses):
                if np.isfinite(_val_loss):
                    self.eval_hist_configs[r].append(C[_id])
                    # Turn it into a maximization problem.
                    self.eval_hist_perfs[r].append(-_val_loss)
            self.logger.info('Evaluation took %.2f seconds' % (time.time() - _start_time))

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
        archs, reduced_archs = [config['estimator'] for config in C], list()
        # Preserve the partial-relationship order.
        for _arch in archs:
            if _arch not in reduced_archs:
                reduced_archs.append(_arch)
        return reduced_archs

    def select_network_architectures(self, algorithm_candidates, dl_evaluator, num_arch=1, **kwargs):
        if len(algorithm_candidates) == 1:
            return algorithm_candidates

        _archs = algorithm_candidates.copy()
        if self.n_jobs > 1:
            # self.executor = ParallelProcessEvaluator(dl_evaluator, n_worker=self.n_jobs)
            with ParallelProcessEvaluator(dl_evaluator, n_worker=self.n_jobs) as executor:
                self.logger.info('Create parallel executor with n_jobs=%d' % self.n_jobs)
                while len(_archs) > num_arch:
                    _archs = self.exec_SEE(_archs, executor=executor)
        else:
            self.nas_evaluator = dl_evaluator
            while len(_archs) > num_arch:
                _archs = self.exec_SEE(_archs)

        return _archs
