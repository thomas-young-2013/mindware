import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter

from solnml.components.models.base_model import BaseClassificationModel
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class Logistic_Regression(BaseClassificationModel):
    def __init__(self, C, penalty, solver, tol, max_iter, random_state=None):
        self.C = C
        self.tol = tol
        self.random_state = random_state
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.estimator = None
        self.time_limit = None

        self.n_jobs = 1

    def fit(self, X, Y):
        from sklearn.linear_model import LogisticRegression

        self.C = float(self.C)

        self.estimator = LogisticRegression(random_state=self.random_state,
                                            solver=self.solver,
                                            penalty=self.penalty,
                                            multi_class='ovr',
                                            C=self.C,
                                            tol=self.tol,
                                            max_iter=self.max_iter,
                                            n_jobs=self.n_jobs)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Logistic-Regression',
                'name': 'Logistic Regression Classification',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        C = UniformFloatHyperparameter("C", 0.03125, 10, log=True,
                                       default_value=1.0)
        tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2, default_value=1e-4,
                                         log=True)

        max_iter = UnParametrizedHyperparameter("max_iter", 3000)

        penalty = CategoricalHyperparameter(name="penalty",
                                            choices=["l1", "l2"],
                                            default_value="l2")
        solver = CategoricalHyperparameter(name="solver", choices=["liblinear", "saga"], default_value="liblinear")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, penalty, solver, tol, max_iter])
        return cs
