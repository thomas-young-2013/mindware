import time
import warnings
import numpy as np
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, train_test_split
from automlToolkit.utils.logging_utils import get_logger
from automlToolkit.components.evaluators.base_evaluator import _BaseEvaluator


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
def holdout_validation(reg, scorer, X, y, test_size=0.3, random_state=1):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_size, random_state=9)
        reg.fit(X_train, y_train)
        return scorer(reg, X_test, y_test)


def get_estimator(config):
    from automlToolkit.components.models.regression import _regressors, _addons
    regressor_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    config_['random_state'] = 1
    try:
        estimator = _regressors[regressor_type](**config_)
    except:
        estimator = _addons.components[regressor_type](**config_)
    return regressor_type, estimator


class RegressionEvaluator(_BaseEvaluator):
    def __init__(self, reg_config, scorer=None, data_node=None, name=None,
                 resampling_strategy='holdout', cv=5, seed=1,
                 estimator=None):
        self.reg_config = reg_config
        self.scorer = scorer
        self.data_node = data_node
        self.name = name
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

        config_dict = config.get_dictionary().copy()
        regressor_id, reg = get_estimator(config_dict)
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
        # print('=' * 6 + '>', self.scorer._sign * score)
        fmt_str = '\n' + ' ' * 5 + '==> '
        self.logger.debug('%s%d-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                          (fmt_str, self.eval_id, regressor_id,
                           self.scorer._sign * score, time.time() - start_time, X_train.shape))
        self.eval_id += 1
        if self.name == 'hpo':
            score = 1 - score
        return score
