import warnings
import numpy as np
import time
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def get_onehot_y(encoder, y):
    y_ = np.reshape(y, (len(y), 1))
    return encoder.transform(y_).toarray()


def dl_holdout_validation(estimator, scorer, dataset, random_state=1, **kwargs):
    start_time = time.time()
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        estimator.fit(dataset, **kwargs)
        if 'profile_epoch' in kwargs or 'profile_iter' in kwargs:
            return time.time() - start_time
        else:
            return scorer._sign * estimator.score(dataset, scorer._score_func)
