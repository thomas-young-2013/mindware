import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


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


class Evaluator(object):
    def __init__(self, clf=None, cv=5, seed=1):
        self.cv = cv
        self.seed = seed
        self.eval_id = 0
        if clf is not None:
            self.clf = clf
        else:
            self.clf = RandomForestClassifier(n_estimators=100, random_state=seed)

    def __call__(self, data_node):
        np.random.seed(self.seed)
        start_time = time.time()
        X_train, y_train = data_node.data
        score = cross_validation(self.clf, X_train, y_train, n_fold=self.cv, random_state=self.seed)
        print('Evaluation %d || Score: %.4f || Time cost: %.2f seconds || Shape: %s' %
              (self.eval_id, score, time.time() - start_time, X_train.shape))
        self.eval_id += 1
        return score
