from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter
import numpy as np

from mindware.components.utils.constants import *
from mindware.components.models.base_model import BaseRegressionModel


class LightGBM(BaseRegressionModel):
    def __init__(self, n_estimators, learning_rate, num_leaves, min_child_weight,
                 subsample, colsample_bytree, reg_alpha, reg_lambda, random_state=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree

        self.n_jobs = 1
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        self.estimator = LGBMRegressor(num_leaves=self.num_leaves,
                                       learning_rate=self.learning_rate,
                                       n_estimators=self.n_estimators,
                                       min_child_weight=self.min_child_weight,
                                       subsample=self.subsample,
                                       colsample_bytree=self.colsample_bytree,
                                       reg_alpha=self.reg_alpha,
                                       reg_lambda=self.reg_lambda,
                                       random_state=self.random_state,
                                       n_jobs=self.n_jobs)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LightGBM Regressor',
                'name': 'LightGBM Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500)
            num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 1023, default_value=31)
            learning_rate = UniformFloatHyperparameter("learning_rate", 0.025, 0.3, default_value=0.1, log=True)
            min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
            subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=1)
            colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.5, 1, default_value=1)
            reg_alpha = UniformFloatHyperparameter('reg_alpha', 1e-10, 10, log=True, default_value=1e-10)
            reg_lambda = UniformFloatHyperparameter("reg_lambda", 1e-10, 10, log=True, default_value=1e-10)
            cs.add_hyperparameters([n_estimators, num_leaves, learning_rate, min_child_weight, subsample,
                                    colsample_bytree, reg_alpha, reg_lambda])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_estimators': hp.randint('lgb_n_estimators', 901) + 100,
                     'num_leaves': hp.randint('lgb_num_leaves', 993) + 31,
                     'learning_rate': hp.loguniform('lgb_learning_rate', np.log(0.025), np.log(0.3)),
                     'min_child_weight': hp.randint('lgb_min_child_weight', 10) + 1,
                     'subsample': hp.uniform('lgb_subsample', 0.5, 1),
                     'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.5, 1),
                     'reg_alpha': hp.loguniform('lgb_reg_alpha', np.log(1e-10), np.log(10)),
                     'reg_lambda': hp.loguniform('lgb_reg_lambda', np.log(1e-10), np.log(10))
                     }

            init_trial = {'n_estimators': 500,
                          'num_leaves': 31,
                          'learning_rate': 0.1,
                          'min_child_weight': 1,
                          'subsample': 1,
                          'colsample_bytree': 1,
                          'reg_alpha': 1e-10,
                          'reg_lambda': 1e-10
                          }

            return space
