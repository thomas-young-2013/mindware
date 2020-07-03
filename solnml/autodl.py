import os
import time
import torch
import numpy as np
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.utils.constant import MAX_INT
from solnml.components.utils.constants import IMG_CLS, TEXT_CLS, OBJECT_DET
from solnml.datasets.base_dataset import BaseDataset
from solnml.components.metrics.metric import get_metric
from solnml.utils.logging_utils import setup_logger, get_logger
from solnml.components.ensemble.dl_ensemble.ensemble_bulider import EnsembleBuilder, ensemble_list
from solnml.components.hpo_optimizer import build_hpo_optimizer
from solnml.components.evaluators.dl_evaluator import DLEvaluator
from solnml.components.evaluators.base_dl_evaluator import get_estimator_with_parameters, TopKModelSaver, get_estimator
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space, \
    get_test_transforms

"""
    imbalanced datasets.
    time_limit
"""


class AutoDL(object):
    def __init__(self, time_limit=300,
                 trial_num=None,
                 dataset_name='default_name',
                 task_type=IMG_CLS,
                 metric='acc',
                 include_algorithms=None,
                 ensemble_method='ensemble_selection',
                 ensemble_size=50,
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

    def get_model_config_space(self, estimator_id):
        if estimator_id in self._estimators:
            clf_class = self._estimators[estimator_id]
        elif estimator_id in self._addons.components:
            clf_class = self._addons.components[estimator_id]
        else:
            raise ValueError("Algorithm %s not supported!" % estimator_id)

        cs = clf_class.get_hyperparameter_search_space()
        model = UnParametrizedHyperparameter("estimator", estimator_id)
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

    def fit(self, train_data: BaseDataset):
        _start_time = time.time()

        for estimator_id in self.include_algorithms:
            cs = self.get_model_config_space(estimator_id)
            default_config = cs.get_default_configuration()
            cs.seed(self.seed)

            hpo_evaluator = DLEvaluator(default_config,
                                        self.task_type,
                                        scorer=self.metric,
                                        dataset=train_data,
                                        device=self.device,
                                        seed=self.seed)
            optimizer = build_hpo_optimizer(self.evaluation_type, hpo_evaluator, cs,
                                            output_dir=self.output_dir,
                                            per_run_time_limit=100000,
                                            trials_per_iter=1,
                                            seed=self.seed, n_jobs=self.n_jobs)
            self.solvers[estimator_id] = optimizer
            self.evaluators[estimator_id] = hpo_evaluator

        algorithm_candidates = self.profile_models()

        # Control flow via round robin.
        n_algorithm = len(algorithm_candidates)
        if self.trial_num is None:
            algo_id = 0
            while time.time() <= _start_time + self.time_limit:
                self.solvers[algorithm_candidates[algo_id]].iterate()
                algo_id = (algo_id + 1) % n_algorithm
        else:
            for id in self.trial_num:
                self.solvers[algorithm_candidates[id % n_algorithm]].iterate()

        # Best model id.
        best_scores_ = list()
        for estimator_id in self.include_algorithms:
            if estimator_id in self.solvers:
                solver_ = self.solvers[estimator_id]
                if len(solver_.perfs) > 0:
                    best_scores_.append(np.max(solver_.perfs))
                else:
                    best_scores_.append(-np.inf)
            else:
                best_scores_.append(-np.inf)
        self.best_algo_id = self.include_algorithms[np.argmax(best_scores_)]
        # Best model configuration.
        solver_ = self.solvers[self.best_algo_id]
        inc_idx = np.argmax(solver_.perfs)
        self.best_algo_config = solver_.configs[inc_idx]

        # Skip Ensemble
        if self.task_type == OBJECT_DET:
            return

        if self.ensemble_method is not None:
            stats = self.fetch_ensemble_members(algorithm_candidates)

            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      metric=self.metric,
                                      device=self.device,
                                      output_dir=self.output_dir)
            self.es.fit(data=train_data)

    def fetch_ensemble_members(self, candidate_algorithms):
        stats = dict()
        # a subset of included algorithms.
        stats['include_algorithms'] = candidate_algorithms
        stats['split_seed'] = self.seed

        self.logger.info('Choose basic models for ensemble stage.')
        self.logger.info('algorithm_id, #models')
        for algo_id in stats['include_algorithms']:
            data = dict()
            leap = 2
            model_num, min_model_num = 20, 5

            hpo_eval_dict = self.solvers[algo_id].eval_dict
            topk_configs = [element[0] for element in self.evaluators[algo_id].topk_model_saver.sorted_list]

            intersection_dict = dict()
            for key in hpo_eval_dict:
                if key[1].get_dictionary() in topk_configs:
                    intersection_dict[key] = hpo_eval_dict[key]

            hpo_eval_list = filter(lambda item: item[1] != -np.inf, intersection_dict.items())
            hpo_eval_list = sorted(hpo_eval_list, key=lambda item: item[1], reverse=True)
            model_items = list()

            if len(hpo_eval_list) > 20:
                idxs = np.arange(min_model_num) * leap
                for idx in idxs:
                    model_items.append(hpo_eval_list[idx])
            else:
                model_items.extend(hpo_eval_list[:min_model_num])

            model_configs = [item[0][1] for item in model_items]
            data['model_configs'] = model_configs
            self.logger.info('%s, %d' % (algo_id, len(model_configs)))
            stats[algo_id] = data
        self.logger.info('Preparing basic models finished.')
        return stats

    def refit(self, dataset: BaseDataset):
        # TODO: Bottom API changes
        if self.es is None:
            config_dict = self.best_algo_config.get_dictionary().copy()
            model_path = self.output_dir + TopKModelSaver.get_configuration_id(config_dict) + '.pt'
            # Remove the old models.
            if os.path.exists(model_path):
                os.remove(model_path)

            # Refit the models.
            _, clf = get_estimator(self.task_type, config_dict, device=self.device)
            # TODO: if train ans val are two parts, we need to merge it into one dataset.
            clf.fit(dataset.train_dataset)
            # Save to the disk.
            torch.save(clf.model.state_dict(), model_path)
        else:
            self.es.refit(dataset)

    def predict_proba(self, test_data: BaseDataset, batch_size=1, n_jobs=1):
        if self.es is None:
            if self.task_type == IMG_CLS:
                test_transforms = get_test_transforms(self.best_algo_config)
                test_data.load_test_data(test_transforms)
            else:
                test_data.load_test_data()
            model_ = get_estimator_with_parameters(self.task_type, self.best_algo_config, test_data.test_dataset,
                                                   device=self.device)
            return model_.predict_proba(test_data.test_dataset, batch_size=batch_size)
        else:
            return self.es.predict(test_data)

    def predict(self, test_data: BaseDataset, batch_size=1, n_jobs=1):
        if self.es is None:
            if self.task_type == IMG_CLS:
                test_transforms = get_test_transforms(self.best_algo_config)
                test_data.load_test_data(test_transforms)
            else:
                test_data.load_test_data()
            model_ = get_estimator_with_parameters(self.task_type, self.best_algo_config, test_data.test_dataset,
                                                   device=self.device)
            return model_.predict(test_data.test_dataset, batch_size=batch_size)
        else:
            return np.argmax(self.es.predict(test_data), axis=-1)

    def score(self, test_data: BaseDataset, metric_func=None):
        if metric_func is None:
            metric_func = self.metric
        return metric_func(self, test_data)
