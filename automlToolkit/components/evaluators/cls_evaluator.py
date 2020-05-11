import time
import numpy as np
from sklearn.metrics.scorer import balanced_accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

from automlToolkit.utils.logging_utils import get_logger
from automlToolkit.components.evaluators.base_evaluator import _BaseEvaluator
from automlToolkit.components.evaluators.evaluate_func import holdout_validation, cross_validation, partial_validation


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
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', 4)
    return classifier_type, estimator


class ClassificationEvaluator(_BaseEvaluator):
    def __init__(self, clf_config, scorer=None, data_node=None, name=None,
                 resampling_strategy='cv', resampling_params=None, seed=1):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params
        self.hpo_config = clf_config
        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.data_node = data_node
        self.name = name
        self.seed = seed
        self.eval_id = 0
        self.onehot_encoder = None
        self.logger = get_logger('Evaluator-%s' % self.name)

    def get_fit_params(self, y, estimator):
        from automlToolkit.components.utils.balancing import get_weights
        _init_params, _fit_params = get_weights(
            y, estimator, None, {}, {})
        return _init_params, _fit_params

    def __call__(self, config, **kwargs):
        start_time = time.time()
        if self.name is None:
            raise ValueError('This evaluator has no name/type!')
        assert self.name in ['hpo', 'fe']

        # Prepare configuration.
        np.random.seed(self.seed)
        config = config if config is not None else self.hpo_config

        downsample_ratio = kwargs.get('data_subsample_ratio', 1.0)
        # Prepare data node.
        if 'data_node' in kwargs:
            data_node = kwargs['data_node']
        else:
            data_node = self.data_node

        X_train, y_train = data_node.data

        config_dict = config.get_dictionary().copy()
        # Prepare training and initial params for classifier.
        init_params, fit_params = {}, {}
        if data_node.enable_balance == 1:
            init_params, fit_params = self.get_fit_params(y_train, config['estimator'])
            for key, val in init_params.items():
                config_dict[key] = val

        if data_node.data_balance == 1:
            fit_params['data_balance'] = True

        classifier_id, clf = get_estimator(config_dict)

        if self.onehot_encoder is None:
            self.onehot_encoder = OneHotEncoder(categories='auto')
            y = np.reshape(y_train, (len(y_train), 1))
            self.onehot_encoder.fit(y)

        try:
            if self.resampling_strategy == 'cv':
                if self.resampling_params is None or 'folds' not in self.resampling_params:
                    folds = 5
                else:
                    folds = self.resampling_params['folds']
                score = cross_validation(clf, self.scorer, X_train, y_train,
                                         n_fold=folds,
                                         random_state=self.seed,
                                         if_stratify=True,
                                         onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                  _ThresholdScorer) else None,
                                         fit_params=fit_params)
            elif self.resampling_strategy == 'holdout':
                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']
                score = holdout_validation(clf, self.scorer, X_train, y_train,
                                           test_size=test_size,
                                           random_state=self.seed,
                                           if_stratify=True,
                                           onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                    _ThresholdScorer) else None,
                                           fit_params=fit_params)
            elif self.resampling_strategy == 'partial':
                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']
                score = partial_validation(clf, self.scorer, X_train, y_train, downsample_ratio,
                                           test_size=test_size,
                                           random_state=self.seed,
                                           if_stratify=True,
                                           onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                    _ThresholdScorer) else None,
                                           fit_params=fit_params)
            else:
                raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)
        except Exception as e:
            if self.name == 'fe':
                raise e
            self.logger.info('%s-evaluator: %s' % (self.name, str(e)))
            score = 0.

        fmt_str = '\n' + ' ' * 5 + '==> '
        self.logger.debug('%s%d-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                          (fmt_str, self.eval_id, classifier_id,
                           score, time.time() - start_time, X_train.shape))
        self.eval_id += 1

        if self.name == 'hpo':
            # Turn it into a minimization problem.
            score = 1. - score
        return score
