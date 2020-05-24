import time

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, \
    CategoricalHyperparameter

from solnml.components.models.base_model import BaseRegressionModel, IterativeComponentWithSampleWeight
from solnml.components.utils.configspace_utils import check_none
from solnml.components.utils.constants import DENSE, UNSIGNED_DATA, PREDICTIONS


class GradientBoostingRegressor(IterativeComponentWithSampleWeight, BaseRegressionModel):
    def __init__(self, loss, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_depth, criterion, max_features,
                 max_leaf_nodes, min_impurity_decrease, random_state=None,
                 verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.fully_fit_ = False
        self.start_time = time.time()
        self.time_limit = None

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):

        from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as GBR
        # Special fix for gradient boosting!
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=X.dtype)
        if refit:
            self.estimator = None

        if self.estimator is None:
            self.learning_rate = float(self.learning_rate)
            self.n_estimators = int(self.n_estimators)
            self.subsample = float(self.subsample)
            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            self.max_features = float(self.max_features)
            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.min_impurity_decrease = float(self.min_impurity_decrease)
            self.verbose = int(self.verbose)

            self.estimator = GBR(
                loss=self.loss,
                learning_rate=self.learning_rate,
                n_estimators=n_iter,
                subsample=self.subsample,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                criterion=self.criterion,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=True,
            )

        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(self.estimator.n_estimators,
                                              self.n_estimators)

        self.estimator.fit(X, y, sample_weight=sample_weight)

        # Apparently this if is necessary
        if self.estimator.n_estimators >= self.n_estimators:
            self.fully_fit_ = True

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
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            loss = CategoricalHyperparameter("loss", ['ls', 'lad'], default_value='ls')
            learning_rate = UniformFloatHyperparameter(
                name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
            n_estimators = UniformIntegerHyperparameter(
                "n_estimators", 50, 500, default_value=200)
            max_depth = UniformIntegerHyperparameter(
                name="max_depth", lower=1, upper=10, default_value=3)
            criterion = CategoricalHyperparameter(
                'criterion', ['friedman_mse', 'mse', 'mae'],
                default_value='friedman_mse')
            min_samples_split = UniformIntegerHyperparameter(
                name="min_samples_split", lower=2, upper=20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                name="min_samples_leaf", lower=1, upper=20, default_value=1)
            min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
            subsample = UniformFloatHyperparameter(
                name="subsample", lower=0.1, upper=1.0, default_value=1.0)
            max_features = UniformFloatHyperparameter(
                "max_features", 0.1, 1.0, default_value=1)
            max_leaf_nodes = UnParametrizedHyperparameter(
                name="max_leaf_nodes", value="None")
            min_impurity_decrease = UnParametrizedHyperparameter(
                name='min_impurity_decrease', value=0.0)
            cs.add_hyperparameters([loss, learning_rate, n_estimators, max_depth,
                                    criterion, min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, subsample,
                                    max_features, max_leaf_nodes,
                                    min_impurity_decrease])

            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'loss': hp.choice('gb_loss', ["ls", "lad"]),
                     'learning_rate': hp.loguniform('gb_learning_rate', np.log(0.01), np.log(1)),
                     # 'n_estimators': hp.randint('gb_n_estimators', 451) + 50,
                     'n_estimators': hp.choice('gb_n_estimators', [100]),
                     'max_depth': hp.randint('gb_max_depth', 8) + 1,
                     'criterion': hp.choice('gb_criterion', ['friedman_mse', 'mse', 'mae']),
                     'min_samples_split': hp.randint('gb_min_samples_split', 19) + 2,
                     'min_samples_leaf': hp.randint('gb_min_samples_leaf', 20) + 1,
                     'min_weight_fraction_leaf': hp.choice('gb_min_weight_fraction_leaf', [0]),
                     'subsample': hp.uniform('gb_subsample', 0.1, 1),
                     'max_features': hp.uniform('gb_max_features', 0.1, 1),
                     'max_leaf_nodes': hp.choice('gb_max_leaf_nodes', [None]),
                     'min_impurity_decrease': hp.choice('gb_min_impurity_decrease', [0])}

            init_trial = {'loss': "ls", 'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3,
                          'criterion': "friedman_mse", 'min_samples_split': 2, 'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0, 'subsample': 1, 'max_features': 1,
                          'max_leaf_nodes': None, 'min_impurity_decrease': 0}
            return space
