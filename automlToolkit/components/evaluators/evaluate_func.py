import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from automlToolkit.components.utils.balancing import get_data


def get_onehot_y(encoder, y):
    y_ = np.reshape(y, (len(y), 1))
    return encoder.transform(y_).toarray()


@ignore_warnings(category=ConvergenceWarning)
def cross_validation(estimator, scorer, X, y, n_fold=5, shuffle=True, fit_params=None, if_stratify=True,
                     onehot=None, random_state=1):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        if if_stratify:
            kfold = StratifiedKFold(n_splits=n_fold, random_state=random_state, shuffle=shuffle)
        else:
            kfold = KFold(n_splits=n_fold, random_state=random_state, shuffle=shuffle)
        scores = list()
        for train_idx, valid_idx in kfold.split(X, y):
            train_x, valid_x = X[train_idx], X[valid_idx]
            train_y, valid_y = y[train_idx], y[valid_idx]
            _fit_params = dict()
            if fit_params:
                if 'sample_weight' in fit_params:
                    _fit_params['sample_weight'] = fit_params['sample_weight'][train_idx]
                elif 'data_balance' in fit_params:
                    X_train, y_train = get_data(X_train, y_train)
            estimator.fit(train_x, train_y, **_fit_params)
            if onehot is not None:
                valid_y = get_onehot_y(onehot, valid_y)
            scores.append(scorer(estimator, valid_x, valid_y))
        return np.mean(scores)


@ignore_warnings(category=ConvergenceWarning)
def holdout_validation(estimator, scorer, X, y, test_size=0.33, fit_params=None, if_stratify=True, onehot=None,
                       random_state=1):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        if if_stratify:
            ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        else:
            ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in ss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            _fit_params = dict()
            if fit_params:
                if 'sample_weight' in fit_params:
                    _fit_params['sample_weight'] = fit_params['sample_weight'][train_index]
                elif 'data_balance' in fit_params:
                    X_train, y_train = get_data(X_train, y_train)
            estimator.fit(X_train, y_train, **_fit_params)
            if onehot is not None:
                y_test = get_onehot_y(onehot, y_test)
            return scorer(estimator, X_test, y_test)


@ignore_warnings(category=ConvergenceWarning)
def partial_validation(estimator, scorer, X, y, data_subsample_ratio, test_size=0.33, fit_params=None, if_stratify=True,
                       onehot=None, random_state=1):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        if if_stratify:
            ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        else:
            ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in ss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            _fit_params = dict()
            if data_subsample_ratio == 1:
                _X_train, _y_train = X_train, y_train
            else:
                if if_stratify:
                    down_ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                else:
                    down_ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                for _, _test_index in down_ss.split(X_train, y_train):
                    _X_train, _y_train = X_train[_test_index], y_train[_test_index]
                    if fit_params:
                        if 'sample_weight' in fit_params:
                            _fit_params['sample_weight'] = fit_params['sample_weight'][train_index]
                        elif 'data_balance' in fit_params:
                            X_train, y_train = get_data(X_train, y_train)

            estimator.fit(_X_train, _y_train, **_fit_params)
            if onehot is not None:
                y_test = get_onehot_y(onehot, y_test)
            return scorer(estimator, X_test, y_test)
