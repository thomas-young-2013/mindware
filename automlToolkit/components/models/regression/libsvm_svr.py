from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter
import numpy as np

from automlToolkit.components.utils.constants import *
from automlToolkit.components.utils.configspace_utils import check_for_bool
from automlToolkit.components.models.base_model import BaseRegressionModel


class LibSVM_SVR(BaseRegressionModel):
    def __init__(self, epsilon, C, kernel, gamma, shrinking, tol, max_iter,
                 degree=3, coef0=0, random_state=None):
        self.epsilon = epsilon
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from sklearn.svm import SVR

        # Nested kernel
        if isinstance(self.kernel, tuple):
            nested_kernel = self.kernel
            self.kernel = nested_kernel[0]
            if self.kernel == 'poly':
                self.degree = nested_kernel[1]['degree']
                self.coef0 = nested_kernel[1]['coef0']
            elif self.kernel == 'sigmoid':
                self.coef0 = nested_kernel[1]['coef0']

        self.epsilon = float(self.epsilon)
        self.C = float(self.C)
        if self.degree is None:
            self.degree = 3
        else:
            self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)
        self.tol = float(self.tol)
        self.max_iter = float(self.max_iter)

        self.shrinking = check_for_bool(self.shrinking)

        self.estimator = SVR(epsilon=self.epsilon,
                             C=self.C,
                             kernel=self.kernel,
                             degree=self.degree,
                             gamma=self.gamma,
                             coef0=self.coef0,
                             shrinking=self.shrinking,
                             tol=self.tol,
                             max_iter=self.max_iter)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LibSVM-SVR',
                'name': 'LibSVM Support Vector Regression',
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
            epsilon = CategoricalHyperparameter("epsilon", [1e-4, 1e-3, 1e-2, 1e-1, 1], default_value=1e-4)
            C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                                           default_value=1.0)
            # No linear kernel here, because we have liblinear
            kernel = CategoricalHyperparameter(name="kernel",
                                               choices=["rbf", "poly", "sigmoid"],
                                               default_value="rbf")
            degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
            gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                               log=True, default_value=0.1)
            coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
            # probability is no hyperparameter, but an argument to the SVM algo
            shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                                  default_value="True")
            tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                             log=True)
            # cache size is not a hyperparameter, but an argument to the program!
            max_iter = UnParametrizedHyperparameter("max_iter", 2000)

            cs = ConfigurationSpace()
            cs.add_hyperparameters([epsilon, C, kernel, degree, gamma, coef0,
                                    shrinking, tol, max_iter])

            degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
            coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
            cs.add_condition(degree_depends_on_poly)
            cs.add_condition(coef0_condition)

            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            coef0 = hp.uniform("libsvm_coef0", -1, 1)
            space = {'C': hp.loguniform('libsvm_C', np.log(0.03125), np.log(32768)),
                     'gamma': hp.loguniform('libsvm_gamma', np.log(3.0517578125e-5), np.log(8)),
                     'shrinking': hp.choice('libsvm_shrinking', ["True", "False"]),
                     'tol': hp.loguniform('libsvm_tol', np.log(1e-5), np.log(1e-1)),
                     'max_iter': hp.choice('libsvm_max_iter', [2000]),
                     'kernel': hp.choice('libsvm_kernel',
                                         [("poly", {'degree': hp.randint('libsvm_degree', 4) + 2, 'coef0': coef0}),
                                          ("rbf", {}),
                                          ("sigmoid", {'coef0': coef0})])}

            init_trial = {'C': 1,
                          'gamma': 0.1,
                          'shrinking': "True",
                          'tol': 1e-3,
                          'max_iter': 2000,
                          'kernel': ("rbf", {})}

            return space
