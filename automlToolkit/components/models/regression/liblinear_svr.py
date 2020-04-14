from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
import numpy as np

from automlToolkit.components.utils.constants import *
from automlToolkit.components.utils.configspace_utils import check_for_bool
from automlToolkit.components.models.base_model import BaseRegressionModel


class LibLinear_SVR(BaseRegressionModel):
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, epsilon, loss, dual, tol, C,
                 fit_intercept, intercept_scaling,
                 random_state=None):
        self.epsilon = epsilon
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from sklearn.svm import LinearSVR

        # In case of nested loss
        if isinstance(self.loss, dict):
            combination = self.loss
            self.loss = combination['loss']
            self.dual = combination['dual']

        self.epsilon = float(self.epsilon)
        self.C = float(self.C)
        self.tol = float(self.tol)

        self.dual = check_for_bool(self.dual)

        self.fit_intercept = check_for_bool(self.fit_intercept)

        self.intercept_scaling = float(self.intercept_scaling)

        self.estimator = LinearSVR(epsilon=self.epsilon,
                                   loss=self.loss,
                                   dual=self.dual,
                                   tol=self.tol,
                                   C=self.C,
                                   fit_intercept=self.fit_intercept,
                                   intercept_scaling=self.intercept_scaling,
                                   random_state=self.random_state)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Liblinear-SVR',
                'name': 'Liblinear Support Vector Regression',
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
            epsilon = CategoricalHyperparameter("epsilon", [1e-4, 1e-3, 1e-2, 1e-1, 1], default_value=1e-4)
            loss = CategoricalHyperparameter(
                "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"], default_value="epsilon_insensitive")
            dual = CategoricalHyperparameter("dual", ['True', 'False'], default_value='True')
            tol = UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
            C = UniformFloatHyperparameter(
                "C", 0.03125, 32768, log=True, default_value=1.0)
            fit_intercept = Constant("fit_intercept", "True")
            intercept_scaling = Constant("intercept_scaling", 1)
            cs.add_hyperparameters([epsilon, loss, dual, tol, C,
                                    fit_intercept, intercept_scaling])

            dual_and_loss = ForbiddenAndConjunction(
                ForbiddenEqualsClause(dual, "False"),
                ForbiddenEqualsClause(loss, "epsilon_insensitive")
            )
            cs.add_forbidden_clause(dual_and_loss)
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'loss': hp.choice('liblinear_combination', [{'loss': "epsilon_insensitive", 'dual': "True"},
                                                                 {'loss': "squared_epsilon_insensitive",
                                                                  'dual': "True"},
                                                                 {'loss': "squared_epsilon_insensitive",
                                                                  'dual': "False"}]),
                     'dual': None,
                     'tol': hp.loguniform('liblinear_tol', np.log(1e-5), np.log(1e-1)),
                     'C': hp.loguniform('liblinear_C', np.log(0.03125), np.log(32768)),
                     'fit_intercept': hp.choice('liblinear_fit_intercept', ["True"]),
                     'intercept_scaling': hp.choice('liblinear_intercept_scaling', [1])}

            init_trial = {'loss': {'loss': "epsilon_insensitive", 'dual': "True"},
                          'tol': 1e-4,
                          'C': 1,
                          'fit_intercept': "True",
                          'intercept_scaling': 1}

            return space
