import time
import numpy as np
from automlToolkit.utils.logging_utils import get_logger
from automlToolkit.components.evaluators.base_evaluator import _BaseEvaluator
from automlToolkit.components.evaluators.evaluate_func import holdout_validation, cross_validation, partial_validation


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
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', 4)
    return regressor_type, estimator


class RegressionEvaluator(_BaseEvaluator):
    def __init__(self, reg_config, scorer=None, data_node=None, name=None,
                 resampling_strategy='holdout', resampling_params=None, seed=1,
                 estimator=None):
        self.reg_config = reg_config
        self.scorer = scorer
        self.data_node = data_node
        self.name = name
        self.estimator = estimator
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params
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

        downsample_ratio = kwargs.get('data_subsample_ratio', 1.0)

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
                if self.resampling_params is None or 'folds' not in self.resampling_params:
                    folds = 5
                else:
                    folds = self.resampling_params['folds']
                score = cross_validation(reg, self.scorer, X_train, y_train,
                                         n_fold=folds,
                                         random_state=self.seed,
                                         if_stratify=False)
            elif self.resampling_strategy == 'holdout':
                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']
                score = holdout_validation(reg, self.scorer, X_train, y_train,
                                           test_size=test_size,
                                           random_state=self.seed,
                                           if_stratify=False)
            elif self.resampling_strategy == 'partial':
                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']
                score = partial_validation(reg, self.scorer, X_train, y_train, downsample_ratio,
                                           test_size=test_size,
                                           random_state=self.seed,
                                           if_stratify=False)
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
