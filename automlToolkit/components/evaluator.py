import time
import warnings
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from automlToolkit.utils.logging_utils import get_logger
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def cross_validation(clf, X, y, n_fold=5, shuffle=True, random_state=1):
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
            clf.fit(train_x, train_y)
            pred = clf.predict(valid_x)
            scores.append(accuracy_score(pred, valid_y))
        return np.mean(scores)


@ignore_warnings(category=ConvergenceWarning)
def holdout_validation(clf, X, y, test_size=0.2, random_state=1):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=test_size, random_state=random_state, stratify=y)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return accuracy_score(y_test, y_pred)


def get_estimator(config):
    from autosklearn.pipeline.components.classification import _classifiers
    classifier_type = config['estimator']
    config_ = config.get_dictionary().copy()
    config_.pop('estimator', None)
    config_['random_state'] = 1
    estimator = _classifiers[classifier_type](**config_)
    return classifier_type, estimator


class Evaluator(object):
    def __init__(self, clf_config, data_node=None, name=None,
                 resampling_strategy='cv', cv=5, seed=1):
        self.clf_config = clf_config
        self.data_node = data_node
        self.name = name
        self.resampling_strategy = resampling_strategy
        self.cv = cv
        self.seed = seed
        self.eval_id = 0
        self.logger = get_logger('Evaluator-%s' % self.name)

    def __call__(self, config, **kwargs):
        start_time = time.time()
        if self.name is None:
            raise ValueError('This evaluator has no name/type!')
        assert self.name in ['hpo', 'fe']

        np.random.seed(self.seed)
        config = config if config is not None else self.clf_config
        classifier_id, clf = get_estimator(config)

        if 'data_node' in kwargs:
            data_node = kwargs['data_node']
        else:
            data_node = self.data_node

        X_train, y_train = data_node.data
        try:
            if self.resampling_strategy == 'cv':
                score = cross_validation(clf, X_train, y_train, n_fold=self.cv, random_state=self.seed)
            elif self.resampling_strategy == 'holdout':
                score = holdout_validation(clf, X_train, y_train, random_state=self.seed)
            else:
                raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)
        except Exception as e:
            self.logger.info('%s-evaluator: %s' % (self.name, str(e)))
            score = 0. if self.name == 'fe' else 1.0

        fmt_str = '\n'+' '*5 + '==> '
        self.logger.debug('%s%d-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                          (fmt_str, self.eval_id, classifier_id,
                           score, time.time() - start_time, X_train.shape))
        self.eval_id += 1

        if self.name == 'hpo':
            # Turn it into a minimization problem.
            score = 1. - score
        return score
