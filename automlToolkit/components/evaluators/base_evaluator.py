import warnings
import numpy as np
from abc import ABCMeta
from collections.abc import Iterable
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, train_test_split
from automlToolkit.components.metrics.metric import get_metric
from automlToolkit.components.utils.constants import *


@ignore_warnings(category=ConvergenceWarning)
def cross_validation(reg, scorer, X, y, n_fold=5, shuffle=True, random_state=1):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        kfold = KFold(n_splits=n_fold, random_state=1, shuffle=shuffle)
        scores = list()
        for train_idx, valid_idx in kfold.split(X, y):
            train_x, train_y, valid_x, valid_y = X[train_idx], y[train_idx], X[valid_idx], y[valid_idx]
            reg.fit(train_x, train_y)
            scores.append(scorer(reg, valid_x, valid_y))
        return np.mean(scores)


@ignore_warnings(category=ConvergenceWarning)
def holdout_validation(estimator, scorer, X, y, train_size=0.3, random_state=1):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=train_size, random_state=random_state)
        return scorer(estimator, X_test, y_test)


def fetch_predict_estimator(task_type, config, X_train, y_train):
    # Build the ML estimator.
    # TODO: check this in future.
    from automlToolkit.components.utils.balancing import get_weights
    _init_params, _fit_params = get_weights(
        y_train, config['estimator'], None, {}, {})
    config_dict = config.get_dictionary().copy()
    for key, val in _init_params.items():
        config_dict[key] = val

    if task_type in CLS_TASKS:
        from automlToolkit.components.evaluators.cls_evaluator import get_estimator
    else:
        from automlToolkit.components.evaluators.reg_evaluator import get_estimator
    _, estimator = get_estimator(config_dict)

    estimator.fit(X_train, y_train, **_fit_params)
    return estimator


class _BaseEvaluator(metaclass=ABCMeta):
    def __init__(self, estimator, metric, task_type,
                 evaluation_strategy, **evaluation_params):
        self.estimator = estimator
        if task_type not in TASK_TYPES:
            raise ValueError('Unsupported task type: %s' % task_type)
        self.metric = get_metric(metric)
        self.evaluation_strategy = evaluation_strategy
        self.evaluation_params = evaluation_params

        if self.evaluation_strategy == 'holdout':
            if 'train_size' not in self.evaluation_params:
                self.evaluation_params['train_size']

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
