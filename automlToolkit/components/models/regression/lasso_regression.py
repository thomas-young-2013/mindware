import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from hyperopt import hp

from automlToolkit.components.models.base_model import BaseRegressionModel
from automlToolkit.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class LassoRegressor(BaseRegressionModel):
    def __init__(self, alpha, tol, max_iter, random_state=None):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y):
        from sklearn.linear_model import Lasso
        self.estimator = Lasso(alpha=self.alpha,
                               tol=self.tol,
                               max_iter=self.max_iter,
                               random_state=self.random_state)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Lasso-Regression',
                'name': 'Lasso Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            alpha = UniformFloatHyperparameter("alpha", 0.01, 32, log=True, default_value=1.0)
            tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2, default_value=1e-4,
                                             log=True)

            max_iter = UniformFloatHyperparameter("max_iter", 100, 1000, q=100, default_value=100)

            cs = ConfigurationSpace()
            cs.add_hyperparameters([alpha, tol, max_iter])
            return cs
        elif optimizer == 'tpe':
            space = {'alpha': hp.loguniform('lasso_alpha', np.log(0.01), np.log(32)),
                     'tol': hp.loguniform('lasso_tol', np.log(1e-6), np.log(1e-2)),
                     'max_iter': hp.uniform('lasso_max_iter', 100, 1000)}

            init_trial = {'alpha': 1, 'tol': 1e-4, 'max_iter': 100}

            return space
