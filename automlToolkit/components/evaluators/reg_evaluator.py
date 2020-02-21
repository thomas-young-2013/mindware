import time
import warnings
import numpy as np
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from automlToolkit.utils.logging_utils import get_logger
from automlToolkit.components.metrics.metric import fetch_scorer


@ignore_warnings(category=ConvergenceWarning)
def cross_validation(reg, scorer, X, y, n_fold=5, shuffle=True, random_state=1):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        kfold = StratifiedKFold(n_splits=n_fold, random_state=1, shuffle=shuffle)
        scores = list()
        for train_idx, valid_idx in kfold.split(X, y):
            train_x = X[train_idx]
            valid_x = X[valid_idx]
            train_y = y[train_idx]
            valid_y = y[valid_idx]
            reg.fit(train_x, train_y)
            pred = reg.predict(valid_x)
            scores.append(scorer(pred, valid_y))
        return np.mean(scores)


@ignore_warnings(category=ConvergenceWarning)
def holdout_validation(reg, scorer, X, y, test_size=0.2, random_state=1):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            return scorer(y_test, y_pred)


def get_estimator(config):
    from autosklearn.pipeline.components.regression import _regressors
    regressor_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    # config_['random_state'] = 1
    estimator = _regressors[regressor_type](**config_)
    return regressor_type, estimator


class RegressionEvaluator(object):
    def __init__(self, reg_config, scorer, data_node=None, name=None,
                 resampling_strategy='cv', cv=5, seed=1,
                 estimator=None):
        self.reg_config = reg_config
        self.scorer = fetch_scorer(scorer, 'regression')
        self.data_node = data_node
        self.name = name
        self.scorer = scorer
        self.estimator = estimator
        self.resampling_strategy = resampling_strategy
        self.cv = cv
        self.seed = seed
        self.eval_id = 0
        self.logger = get_logger('RegressionEvaluator-%s' % self.name)

    def __call__(self, config, **kwargs):
        start_time = time.time()
        if self.name is None:
            raise ValueError('This evaluator has no name/type!')
        assert self.name in ['hpo', 'fe']

        # Prepare configuration.
        np.random.seed(self.seed)
        config = config if config is not None else self.reg_config

        # Prepare data node.
        if 'data_node' in kwargs:
            data_node = kwargs['data_node']
        else:
            data_node = self.data_node

        X_train, y_train = data_node.data
        if self.estimator is None:
            config_dict = config.get_dictionary().copy()
            regressor_id, reg = get_estimator(config_dict)
        else:
            reg = self.estimator
            regressor_id = self.estimator.__class__.__name__

        try:
            if self.resampling_strategy == 'cv':
                score = cross_validation(reg, self.scorer, X_train, y_train,
                                         n_fold=self.cv, random_state=self.seed)
            elif self.resampling_strategy == 'holdout':
                score = holdout_validation(reg, self.scorer, X_train, y_train,
                                           random_state=self.seed)
            else:
                raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)
        except Exception as e:
            if self.name == 'fe':
                raise e
            self.logger.info('%s-evaluator: %s' % (self.name, str(e)))
            return np.inf

        fmt_str = '\n'+' '*5 + '==> '
        self.logger.debug('%s%d-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                          (fmt_str, self.eval_id, regressor_id,
                           score, time.time() - start_time, X_train.shape))
        self.eval_id += 1
        if self.name == 'hpo':
            score = -score
        return score
