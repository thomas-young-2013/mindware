import time
import warnings
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.scorer import balanced_accuracy_scorer
from sklearn.model_selection import train_test_split

from automlToolkit.utils.logging_utils import get_logger
from automlToolkit.components.evaluators.base_evaluator import _BaseEvaluator


@ignore_warnings(category=ConvergenceWarning)
def partial_validation(clf, scorer, X, y, data_subsample_ratio, fit_params=None, random_state=1):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=1)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            _fit_params = dict()
            if len(fit_params) > 0:
                _fit_params['sample_weight'] = fit_params['sample_weight'][train_index]

            _, _X_train, _, _y_train = train_test_split(X_train, y_test,
                                                        test_size=data_subsample_ratio,
                                                        stratify=y_train,
                                                        random_state=random_state)
            clf.fit(_X_train, _y_train, **_fit_params)
            return scorer(clf, X_test, y_test)


def get_estimator(config):
    from automlToolkit.components.models.classification import _classifiers, _addons
    classifier_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    config_['random_state'] = 1
    try:
        estimator = _classifiers[classifier_type](**config_)
    except:
        estimator = _addons.components[classifier_type](**config_)
    return classifier_type, estimator


class MfseEvaluator(_BaseEvaluator):
    def __init__(self, clf_config, scorer=None, data_node=None, name=None, seed=1):
        self.clf_config = clf_config
        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.data_node = data_node
        self.name = name
        self.seed = seed
        self.logger = get_logger('MfseEvaluator-%s' % self.name)
        self.init_params = None
        self.fit_params = None

    def get_fit_params(self, y, estimator):
        from automlToolkit.components.utils.balancing import get_weights
        _init_params, _fit_params = get_weights(
            y, estimator, None, {}, {})
        self.init_params = _init_params
        self.fit_params = _fit_params

    def __call__(self, config, **kwargs):
        start_time = time.time()
        if self.name is None:
            raise ValueError('This evaluator has no name/type!')
        assert self.name in ['hpo', 'fe']

        # Prepare configuration.
        np.random.seed(self.seed)
        config = config if config is not None else self.clf_config

        assert 'data_subsample_ratio' in kwargs
        test_size = kwargs['data_subsample_ratio']

        # Prepare data node.
        if 'data_node' in kwargs:
            data_node = kwargs['data_node']
        else:
            data_node = self.data_node

        X_train, y_train = data_node.data

        # Prepare training and initial params for classifier.
        if data_node.enable_balance or True:
            if self.init_params is None or self.fit_params is None:
                self.get_fit_params(y_train, config['estimator'])

        config_dict = config.get_dictionary().copy()
        for key, val in self.init_params.items():
            config_dict[key] = val

        classifier_id, clf = get_estimator(config_dict)

        try:
            score = partial_validation(clf, self.scorer, X_train, y_train, test_size,
                                       random_state=self.seed, fit_params=self.fit_params)
        except Exception as e:
            if self.name == 'fe':
                raise e
            self.logger.info('%s-evaluator: %s' % (self.name, str(e)))
            score = 0.

        fmt_str = '\n' + ' ' * 5 + '==> '
        self.logger.debug('%s-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                          (fmt_str, classifier_id,
                           score, time.time() - start_time, X_train.shape))

        if self.name == 'hpo':
            # Turn it into a minimization problem.
            score = 1. - score
        return score
