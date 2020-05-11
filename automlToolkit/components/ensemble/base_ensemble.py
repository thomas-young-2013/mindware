from sklearn.metrics.scorer import _BaseScorer
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import os
import numpy as np
import pickle as pkl
import time

from automlToolkit.components.utils.constants import CLS_TASKS
from automlToolkit.components.evaluators.base_evaluator import fetch_predict_estimator
from automlToolkit.components.ensemble.unnamed_ensemble import choose_base_models_classification, \
    choose_base_models_regression


class BaseEnsembleModel(object):
    """Base class for model ensemble"""

    def __init__(self, stats, ensemble_method: str,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None):
        self.stats = stats
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.output_dir = output_dir

        self.train_predictions = []
        self.config_list = []
        self.train_data_dict = {}
        self.train_labels = None
        self.seed = self.stats['split_seed']
        self.timestamp = str(time.time())
        for algo_id in self.stats["include_algorithms"]:
            model_to_eval = self.stats[algo_id]['model_to_eval']
            for idx, (node, config) in enumerate(model_to_eval):
                X, y = node.data

                # TODO: Hyperparameter
                test_size = 0.33

                if self.task_type in CLS_TASKS:
                    ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                else:
                    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)

                for train_index, test_index in ss.split(X, y):
                    X_train, X_valid = X[train_index], X[test_index]
                    y_train, y_valid = y[train_index], y[test_index]

                if self.train_labels is not None:
                    assert (self.train_labels == y_valid).all()
                else:
                    self.train_labels = y_valid

                estimator = fetch_predict_estimator(self.task_type, config, X_train, y_train,
                                                    weight_balance=node.enable_balance,
                                                    data_balance=node.data_balance
                                                    )
                if self.task_type in CLS_TASKS:
                    y_valid_pred = estimator.predict_proba(X_valid)
                else:
                    y_valid_pred = estimator.predict(X_valid)
                self.train_predictions.append(y_valid_pred)
        if len(self.train_predictions) < self.ensemble_size:
            self.ensemble_size = len(self.train_predictions)

        if ensemble_method == 'ensemble_selection':
            return

        if task_type in CLS_TASKS:
            self.base_model_mask = choose_base_models_classification(np.array(self.train_predictions),
                                                                     self.ensemble_size)
        else:
            self.base_model_mask = choose_base_models_regression(np.array(self.train_predictions), np.array(y_valid),
                                                                 self.ensemble_size)
        self.ensemble_size = sum(self.base_model_mask)

    def fit(self, data):
        raise NotImplementedError

    def predict(self, data, solvers):
        raise NotImplementedError
