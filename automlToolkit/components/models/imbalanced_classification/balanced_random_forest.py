import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from automlToolkit.components.models.base_model import BaseClassificationModel
from automlToolkit.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from automlToolkit.components.utils.configspace_utils import check_none, check_for_bool
from automlToolkit.components.utils.constants import *


class BalancedRandomForest(BaseClassificationModel):

    def __init__(self, n_estimators, criterion, max_features,
                 min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, bootstrap,
                 min_impurity_decrease, sampling_strategy, replacement,
                 random_state=None, n_jobs=-1,
                 class_weight=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y, sample_weight=None):
        from imblearn.ensemble import BalancedRandomForestClassifier
        estimator = BalancedRandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            bootstrap=self.bootstrap,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
            sampling_strategy=self.sampling_strategy,
            replacement=self.replacement)

        estimator.fit(X, Y)

        self.estimator = estimator
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
        return {'shortname': 'Bal_RF',
                'name': 'Balanced Random Forest Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            n_estimators = Constant("n_estimators", 100)
            criterion = CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default_value="gini")

            # The maximum number of features used in the forest is calculated as m^max_features, where
            # m is the total number of features, and max_features is the hyperparameter specified below.
            # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
            # corresponds with Geurts' heuristic.
            max_features = UniformFloatHyperparameter(
                "max_features", 0., 1., default_value=0.5)

            min_samples_split = UniformIntegerHyperparameter(
                "min_samples_split", 2, 20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                "min_samples_leaf", 1, 20, default_value=1)
            min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
            min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
            bootstrap = CategoricalHyperparameter(
                "bootstrap", ["True", "False"], default_value="True")
            sampling_strategy = CategoricalHyperparameter(
                name="sampling_strategy", choices=["majority", "not minority", "not majority", "all"],
                default_value="not minority")
            replacement = CategoricalHyperparameter(
                "replacement", ["True", "False"], default_value="False")
            cs.add_hyperparameters([n_estimators, criterion, max_features,
                                    min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf,
                                    bootstrap, min_impurity_decrease, sampling_strategy, replacement])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_estimators': hp.choice('bal_rf_n_estimators', [100]),
                     'criterion': hp.choice('bal_rf_criterion', ["gini", "entropy"]),
                     'max_features': hp.uniform('bal_rf_max_features', 0, 1),
                     'min_samples_split': hp.randint('bal_rf_min_samples_split', 19) + 2,
                     'min_samples_leaf': hp.randint('bal_rf_min_samples_leaf', 20) + 1,
                     'min_weight_fraction_leaf': hp.choice('bal_rf_min_weight_fraction_leaf', [0]),
                     'min_impurity_decrease': hp.choice('bal_rf_min_impurity_decrease', [0]),
                     'bootstrap': hp.choice('bal_rf_bootstrap', ["True", "False"]),
                     'sampling_strategy': hp.choice('bal_rf_sampling_strategy',
                                                    ["majority", "not minority", "not majority", "all"]),
                     'replacement': hp.choice('bal_rf_replacement', ["True", "False"]),
                     }

            init_trial = {'n_estimators': 100,
                          'criterion': "gini",
                          'max_features': 0.5,
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0,
                          'min_impurity_decrease': 0,
                          'bootstrap': "False",
                          'sampling_strategy': "not minority",
                          'replacement': "False"
                          }
            return space
