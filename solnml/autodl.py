import os
import time
import numpy as np
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.components.utils.constants import IMG_CLS
from solnml.datasets.image_dataset import BaseDataset
from solnml.components.metrics.metric import get_metric
from solnml.utils.logging_utils import setup_logger, get_logger
from solnml.components.ensemble import ensemble_list
from solnml.components.ensemble import EnsembleBuilder
from solnml.components.hpo_optimizer import build_hpo_optimizer
from solnml.components.models.img_classification import _classifiers as _img_classifiers
from solnml.components.evaluators.img_cls_evaluator import ImgClassificationEvaluator, TopKModelSaver
from solnml.components.evaluators.base_dl_evaluator import get_estimator_with_parameters
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space

img_classification_algorithms = _img_classifiers.keys()

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
                 output_dir="logs",
                 random_state=1,
                 n_jobs=1):
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
                self.include_algorithms = list(img_classification_algorithms)
            else:
                raise ValueError("Unknown task type %s" % task_type)

        if ensemble_method is not None and ensemble_method not in ensemble_list:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)

        self.solvers = dict()
        self.evaluators = dict()
        # Single model.
        self.best_algo_id = None
        self.best_algo_config = None
        # Ensemble models.
        self.candidate_algo_ids = None

    def _get_logger(self, name):
        logger_name = 'SolnML-%s(%d)' % (name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)

    def fit(self, train_data: BaseDataset):
        _start_time = time.time()
        from solnml.components.models.img_classification import _classifiers, _addons
        for estimator_id in self.include_algorithms:
            if estimator_id in _classifiers:
                clf_class = _classifiers[estimator_id]
            elif estimator_id in _addons.components:
                clf_class = _addons.components[estimator_id]
            else:
                raise ValueError("Algorithm %s not supported!" % estimator_id)

            cs = clf_class.get_hyperparameter_search_space()
            model = UnParametrizedHyperparameter("estimator", estimator_id)
            cs.add_hyperparameter(model)
            config_space = cs
            default_config = cs.get_default_configuration()
            config_space.seed(self.seed)

            aug_space=get_aug_hyperparameter_space()
            cs.add_hyperparameters(aug_space.get_hyperparameters())
            cs.add_conditions(aug_space.get_conditions())

            hpo_evaluator = ImgClassificationEvaluator(default_config, scorer=self.metric,
                                                       dataset=train_data,
                                                       seed=self.seed)
            optimizer = build_hpo_optimizer(self.evaluation_type, hpo_evaluator, cs,
                                            output_dir=self.output_dir,
                                            per_run_time_limit=100000,
                                            trials_per_iter=1,
                                            seed=self.seed, n_jobs=self.n_jobs)

            # Control flow.
            if self.trial_num is None:
                while time.time() <= _start_time + self.time_limit:
                    optimizer.iterate()
            else:
                for _ in self.trial_num:
                    optimizer.iterate()
            self.solvers[estimator_id] = optimizer
            self.evaluators[estimator_id] = hpo_evaluator

        # Best model id.
        best_scores_ = list()
        for estimator_id in self.include_algorithms:
            if estimator_id in self.solvers:
                solver_ = self.solvers[estimator_id]
                best_scores_.append(np.max(solver_.perfs))
            else:
                best_scores_.append(-np.inf)
        self.best_algo_id = self.include_algorithms[np.argmax(best_scores_)]
        # Best model configuration.
        solver_ = self.solvers[self.best_algo_id]
        inc_idx = np.argmax(solver_.perfs)
        self.best_algo_config = solver_.configs[inc_idx]

        if self.ensemble_method is not None:
            # stats = self.fetch_ensemble_members()
            stats = self.fetch_ensemble_members()

            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      metric=self.metric,
                                      output_dir=self.output_dir)
            self.es.fit(data=train_data)

    def fetch_ensemble_members(self):
        stats = dict()
        stats['candidate_algorithms'] = self.include_algorithms

        # a subset of included algorithms.
        stats['include_algorithms'] = self.include_algorithms
        stats['split_seed'] = self.seed

        self.logger.info('Choose basic models for ensemble stage.')
        self.logger.info('algorithm_id, #models')
        for algo_id in stats['include_algorithms']:
            data = dict()
            leap = 2
            model_num, min_model_num = 20, 5

            hpo_eval_dict = self.solvers[algo_id].eval_dict

            hpo_eval_list = sorted(hpo_eval_dict.items(), key=lambda item: item[1], reverse=True)
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

    def refit(self):
        pass

    def predict_proba(self, test_data: BaseDataset):
        pass

    def predict(self, test_data: BaseDataset):
        model_ = get_estimator_with_parameters(self.best_algo_config)
        return model_.predict(test_data)

    def score(self, test_data: BaseDataset, metric_func=None):
        if metric_func is None:
            metric_func = self.metric
        return metric_func(self, test_data)
