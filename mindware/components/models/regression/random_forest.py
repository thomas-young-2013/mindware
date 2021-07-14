from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
import numpy as np

from mindware.components.models.base_model import BaseRegressionModel, IterativeComponentWithSampleWeight
from mindware.components.utils.constants import *
from mindware.components.utils.configspace_utils import check_none, check_for_bool

import time


class RandomForest(
    IterativeComponentWithSampleWeight,
    BaseRegressionModel,
):
    def __init__(self, criterion, max_features,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, bootstrap, max_leaf_nodes,
                 min_impurity_decrease, random_state=None, n_jobs=1):
        self.n_estimators = self.get_max_iter()
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator = None
        self.start_time = time.time()
        self.time_limit = None

    @staticmethod
    def get_max_iter():
        return 100

    def get_current_iter(self):
        return self.estimator.n_estimators

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        from sklearn.ensemble import RandomForestRegressor

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.n_estimators = int(self.n_estimators)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)

            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

            if self.max_features not in ("sqrt", "log2", "auto"):
                max_features = int(X.shape[1] ** float(self.max_features))
            else:
                max_features = self.max_features

            self.bootstrap = check_for_bool(self.bootstrap)

            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)

            self.min_impurity_decrease = float(self.min_impurity_decrease)

            # initial fit of only increment trees
            self.estimator = RandomForestRegressor(
                n_estimators=n_iter,
                criterion=self.criterion,
                max_features=max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                bootstrap=self.bootstrap,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                warm_start=True)
        else:

            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(self.estimator.n_estimators,
                                              self.n_estimators)

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False

        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RF',
                'name': 'Random Forest Regressor',
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
            cs = ConfigurationSpace()
            criterion = CategoricalHyperparameter(
                "criterion", ["mse", "mae"], default_value="mse")

            # The maximum number of features used in the forest is calculated as m^max_features, where
            # m is the total number of features, and max_features is the hyperparameter specified below.
            # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
            # corresponds with Geurts' heuristic.
            max_features = UniformFloatHyperparameter(
                "max_features", 0., 1., default_value=0.5)

            max_depth = UnParametrizedHyperparameter("max_depth", "None")
            min_samples_split = UniformIntegerHyperparameter(
                "min_samples_split", 2, 20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                "min_samples_leaf", 1, 20, default_value=1)
            min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
            max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
            min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
            bootstrap = CategoricalHyperparameter(
                "bootstrap", ["True", "False"], default_value="True")
            cs.add_hyperparameters([criterion, max_features,
                                    max_depth, min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, max_leaf_nodes,
                                    bootstrap, min_impurity_decrease])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'criterion': hp.choice('rf_criterion', ["mse", "mae"]),
                     'max_features': hp.uniform('rf_max_features', 0, 1),
                     'max_depth': hp.choice('rf_max_depth', [None]),
                     'min_samples_split': hp.randint('rf_min_samples_split', 19) + 2,
                     'min_samples_leaf': hp.randint('rf_min_samples_leaf', 20) + 1,
                     'min_weight_fraction_leaf': hp.choice('rf_min_weight_fraction_leaf', [0]),
                     'max_leaf_nodes': hp.choice('rf_max_leaf_nodes', [None]),
                     'min_impurity_decrease': hp.choice('rf_min_impurity_decrease', [0]),
                     'bootstrap': hp.choice('rf_bootstrap', ["True", "False"])}

            init_trial = {'criterion': "mse",
                          'max_features': 0.5,
                          'max_depth': None,
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0,
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0,
                          'bootstrap': "False"}

            return space
