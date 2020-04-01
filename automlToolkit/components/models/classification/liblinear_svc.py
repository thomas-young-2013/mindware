from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
import numpy as np
from hyperopt import hp

from automlToolkit.components.utils.model_util import softmax
from automlToolkit.components.utils.constants import *
from automlToolkit.components.utils.configspace_utils import check_none, check_for_bool
from automlToolkit.components.models.base_model import BaseClassificationModel


class LibLinear_SVC(BaseClassificationModel):
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, penalty, loss, dual, tol, C, multi_class,
                 fit_intercept, intercept_scaling, class_weight=None,
                 random_state=None):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y):
        import sklearn.svm
        import sklearn.multiclass

        # In case of nested penalty
        if isinstance(self.penalty, dict):
            combination = self.penalty
            self.penalty = combination['penalty']
            self.loss = combination['loss']
            self.dual = combination['dual']

        self.C = float(self.C)
        self.tol = float(self.tol)

        self.dual = check_for_bool(self.dual)

        self.fit_intercept = check_for_bool(self.fit_intercept)

        self.intercept_scaling = float(self.intercept_scaling)

        if check_none(self.class_weight):
            self.class_weight = None

        estimator = sklearn.svm.LinearSVC(penalty=self.penalty,
                                          loss=self.loss,
                                          dual=self.dual,
                                          tol=self.tol,
                                          C=self.C,
                                          class_weight=self.class_weight,
                                          fit_intercept=self.fit_intercept,
                                          intercept_scaling=self.intercept_scaling,
                                          multi_class=self.multi_class,
                                          random_state=self.random_state)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        df = self.estimator.decision_function(X)
        return softmax(df)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Liblinear-SVC',
                'name': 'Liblinear Support Vector Classification',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()

            penalty = CategoricalHyperparameter(
                "penalty", ["l1", "l2"], default_value="l2")
            loss = CategoricalHyperparameter(
                "loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
            dual = CategoricalHyperparameter("dual", ['True', 'False'], default_value='True')
            # This is set ad-hoc
            tol = UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
            C = UniformFloatHyperparameter(
                "C", 0.03125, 32768, log=True, default_value=1.0)
            multi_class = Constant("multi_class", "ovr")
            # These are set ad-hoc
            fit_intercept = Constant("fit_intercept", "True")
            intercept_scaling = Constant("intercept_scaling", 1)
            cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class,
                                    fit_intercept, intercept_scaling])

            penalty_and_loss = ForbiddenAndConjunction(
                ForbiddenEqualsClause(penalty, "l1"),
                ForbiddenEqualsClause(loss, "hinge")
            )
            constant_penalty_and_loss = ForbiddenAndConjunction(
                ForbiddenEqualsClause(dual, "False"),
                ForbiddenEqualsClause(penalty, "l2"),
                ForbiddenEqualsClause(loss, "hinge")
            )
            penalty_and_dual = ForbiddenAndConjunction(
                ForbiddenEqualsClause(dual, "True"),
                ForbiddenEqualsClause(penalty, "l1")
            )
            cs.add_forbidden_clause(penalty_and_loss)
            cs.add_forbidden_clause(constant_penalty_and_loss)
            cs.add_forbidden_clause(penalty_and_dual)
            return cs
        elif optimizer == 'tpe':
            space = {'penalty': hp.choice('liblinear_combination',
                                          [{'penalty': "l1", 'loss': "squared_hinge", 'dual': "False"},
                                           {'penalty': "l2", 'loss': "hinge", 'dual': "True"},
                                           {'penalty': "l2", 'loss': "squared_hinge", 'dual': "True"},
                                           {'penalty': "l2", 'loss': "squared_hinge", 'dual': "False"}]),
                     'loss': None,
                     'dual': None,
                     'tol': hp.loguniform('liblinear_tol', np.log(1e-5), np.log(1e-1)),
                     'C': hp.loguniform('liblinear_C', np.log(0.03125), np.log(32768)),
                     'multi_class': hp.choice('liblinear_multi_class', ["ovr"]),
                     'fit_intercept': hp.choice('liblinear_fit_intercept', ["True"]),
                     'intercept_scaling': hp.choice('liblinear_intercept_scaling', [1])}

            init_trial = {'penalty': {'penalty': "l2", 'loss': "squared_hinge", 'dual': "True"},
                          'tol': 1e-4,
                          'C': 1,
                          'multiclass': "ovr",
                          'fit_intercept': "True",
                          'intercept_scaling': 1}

            return space
