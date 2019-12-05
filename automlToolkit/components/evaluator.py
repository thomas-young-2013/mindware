import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from automlToolkit.utils.logging_utils import get_logger


def cross_validation(clf, X, y, n_fold=5, shuffle=True, random_state=1):
    kfold = StratifiedKFold(n_splits=n_fold, random_state=random_state, shuffle=shuffle)
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


def get_estimator(config):
    from autosklearn.pipeline.components.classification import _classifiers
    classifier_type = config['estimator']
    config_ = config.get_dictionary().copy()
    config_.pop('estimator', None)
    estimator = _classifiers[classifier_type](**config_)
    return classifier_type, estimator


class Evaluator(object):
    def __init__(self, clf_config, data_node=None, name=None, cv=5, seed=1):
        self.clf_config = clf_config
        self.data_node = data_node
        self.name = name
        self.cv = cv
        self.seed = seed
        self.eval_id = 0
        self.logger = get_logger('Evaluator-%s' % self.name)

    def __call__(self, config, **kwargs):
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

        start_time = time.time()
        X_train, y_train = data_node.data
        score = cross_validation(clf, X_train, y_train, n_fold=self.cv, random_state=self.seed)

        fmt_str = '\n'+' '*5 + '==> '
        self.logger.debug('%s%d-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                          (fmt_str, self.eval_id, classifier_id,
                           score, time.time() - start_time, X_train.shape))
        self.eval_id += 1

        if self.name == 'hpo':
            # Turn it into a minimization problem.
            score = 1. - score
        return score
