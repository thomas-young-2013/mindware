from sklearn.metrics.scorer import _BaseScorer
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np
import pickle as pkl
import time

from solnml.components.utils.constants import CLS_TASKS
from solnml.components.ensemble.unnamed_ensemble import choose_base_models_classification, \
    choose_base_models_regression
from solnml.components.feature_engineering.parse import construct_node
from solnml.utils.logging_utils import get_logger


class BaseEnsembleModel(object):
    """Base class for model ensemble"""

    def __init__(self, stats, ensemble_method: str,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 data_node,
                 output_dir=None):
        self.stats = stats
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.output_dir = output_dir
        self.node = data_node

        self.predictions = []
        self.train_labels = None
        self.timestamp = str(time.time())
        logger_name = 'EnsembleBuilder'
        self.logger = get_logger(logger_name)

        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):
                with open(path, 'rb')as f:
                    op_list, model = pkl.load(f)
                _node = self.node.copy_()
                _node = construct_node(_node, op_list)

                # TODO: Test size
                test_size = 0.33
                X, y = _node.data

                if self.task_type in CLS_TASKS:
                    ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
                else:
                    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=1)

                for train_index, val_index in ss.split(X, y):
                    X_valid = X[val_index]
                    y_valid = y[val_index]

                if self.train_labels is not None:
                    assert (self.train_labels == y_valid).all()
                else:
                    self.train_labels = y_valid

                if self.task_type in CLS_TASKS:
                    y_valid_pred = model.predict_proba(X_valid)
                else:
                    y_valid_pred = model.predict(X_valid)
                self.predictions.append(y_valid_pred)

        if len(self.predictions) < self.ensemble_size:
            self.ensemble_size = len(self.predictions)

        if ensemble_method == 'ensemble_selection':
            return

        if task_type in CLS_TASKS:
            self.base_model_mask = choose_base_models_classification(np.array(self.predictions),
                                                                     self.ensemble_size)
        else:
            self.base_model_mask = choose_base_models_regression(np.array(self.predictions), np.array(y_valid),
                                                                 self.ensemble_size)
        self.ensemble_size = sum(self.base_model_mask)

    def fit(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def get_ens_model_info(self):
        raise NotImplementedError

    # TODO: Refit
    def refit(self):
        raise NotImplementedError
