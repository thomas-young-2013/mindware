import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from mindware.components.models.base_model import BaseClassificationModel
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from mindware.components.utils.configspace_utils import check_none, check_for_bool


class BalancedBagging(BaseClassificationModel):

    def __init__(self, n_estimators, max_features,
                 max_depth, bootstrap, bootstrap_features,
                 sampling_strategy, replacement, random_state=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y, sample_weight=None):
        import sklearn.tree
        if self.estimator is None:
            self.max_depth = int(self.max_depth)
            self.estimator = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth)
        from imblearn.ensemble import BalancedBaggingClassifier
        estimator = BalancedBaggingClassifier(base_estimator=self.estimator,
                                              n_estimators=self.n_estimators,
                                              max_features=self.max_features,
                                              bootstrap=self.bootstrap,
                                              bootstrap_features=self.bootstrap_features,
                                              sampling_strategy=self.sampling_strategy,
                                              replacement=self.replacement,
                                              n_jobs=self.n_jobs,
                                              random_state=self.random_state)
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
        return {'shortname': 'Bal_Bagging',
                'name': 'Balanced Bagging Classifier',
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
            n_estimators = UniformIntegerHyperparameter(
                name="n_estimators", lower=50, upper=500, default_value=50, log=False)
            max_features = UniformFloatHyperparameter(
                "max_features", 0., 1., default_value=0.5)
            bootstrap = CategoricalHyperparameter(
                "bootstrap", ["True", "False"], default_value="True")
            bootstrap_features = CategoricalHyperparameter(
                "bootstrap_features", ["True", "False"], default_value="False")
            sampling_strategy = CategoricalHyperparameter(
                name="sampling_strategy", choices=["majority", "not minority", "not majority", "all"],
                default_value="not minority")
            replacement = CategoricalHyperparameter(
                "replacement", ["True", "False"], default_value="False")
            max_depth = UniformIntegerHyperparameter(
                name="max_depth", lower=1, upper=10, default_value=1, log=False)
            cs.add_hyperparameters(
                [n_estimators, max_features, bootstrap, bootstrap_features, sampling_strategy, replacement,
                 max_depth])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_estimators': hp.randint('bal_bagging_n_estimators', 451) + 50,
                     'max_features': hp.uniform('bal_bagging_max_features', 0, 1),
                     'bootstrap': hp.choice('bal_bagging_bootstrap', ["True", "False"]),
                     'bootstrap_features': hp.choice('bal_bagging_bootstrap_features', ["True", "False"]),
                     'sampling_strategy': hp.choice('bal_bagging_sampling_strategy',
                                                    ["majority", "not minority", "not majority", "all"]),
                     'replacement': hp.choice('bal_bagging_replacement', ["True", "False"]),
                     'max_depth': hp.randint('bal_bagging_max_depth', 10) + 1}
            init_trial = {'n_estimators': 10,
                          'max_features': 0.5,
                          'bootstrap': "True",
                          'bootstrap_features': "False",
                          'sampling_strategy': "not minority",
                          'replacement': "False",
                          'max_depth': 1}
            return space
