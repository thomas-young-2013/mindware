import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from automlToolkit.components.models.base_model import BaseClassificationModel
from automlToolkit.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from automlToolkit.components.utils.configspace_utils import check_none, check_for_bool


class EasyEnsemble(BaseClassificationModel):

    def __init__(self, n_estimators,
                 sampling_strategy,
                 replacement,
                 ab_n_estimators,
                 ab_max_depth,
                 ab_learning_rate,
                 ab_algorithm,
                 n_jobs=-1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        self.random_state = random_state

        # Parameters for Adaboost base learner
        self.ab_max_depth = ab_max_depth
        self.ab_n_estimators = ab_n_estimators
        self.ab_learning_rate = ab_learning_rate
        self.ab_algorithm = ab_algorithm
        self.n_jobs = n_jobs
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y, sample_weight=None):
        import sklearn.tree
        if self.estimator is None:
            self.ab_max_depth = int(self.ab_max_depth)
            base_estimator = sklearn.tree.DecisionTreeClassifier(max_depth=self.ab_max_depth)
            self.estimator = sklearn.ensemble.AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=self.ab_n_estimators,
                learning_rate=self.ab_learning_rate,
                algorithm=self.ab_algorithm,
                random_state=self.random_state
            )
        from imblearn.ensemble import EasyEnsembleClassifier
        estimator = EasyEnsembleClassifier(base_estimator=self.estimator,
                                           n_estimators=self.n_estimators,
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
        return {'shortname': 'Easy_Ensemble',
                'name': 'Easy Ensemble Classifier',
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
            sampling_strategy = CategoricalHyperparameter(
                name="sampling_strategy", choices=["majority", "not minority", "not majority", "all"],
                default_value="not minority")
            replacement = CategoricalHyperparameter(
                "replacement", ["True", "False"], default_value="False")

            ab_n_estimators = UniformIntegerHyperparameter(
                name="ab_n_estimators", lower=50, upper=500, default_value=50, log=False)
            ab_learning_rate = UniformFloatHyperparameter(
                name="ab_learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
            ab_algorithm = CategoricalHyperparameter(
                name="ab_algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
            ab_max_depth = UniformIntegerHyperparameter(
                name="ab_max_depth", lower=1, upper=10, default_value=1, log=False)
            cs.add_hyperparameters([n_estimators, sampling_strategy, replacement, ab_n_estimators,
                                    ab_learning_rate, ab_algorithm, ab_max_depth])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_estimators': hp.randint('easy_ensemble_n_estimators', 451) + 50,
                     'sampling_strategy': hp.choice('easy_ensemble_sampling_strategy',
                                                    ["majority", "not minority", "not majority", "all"]),
                     'replacement': hp.choice('easy_ensemble_replacement', ["True", "False"]),

                     'ab_n_estimators': hp.randint('ab_n_estimators', 451) + 50,
                     'ab_learning_rate': hp.loguniform('ab_learning_rate', np.log(0.01), np.log(2)),
                     'ab_algorithm': hp.choice('ab_algorithm', ["SAMME.R", "SAMME"]),
                     'ab_max_depth': hp.randint('ab_max_depth', 10) + 1}
            init_trial = {'n_estimators': 10,
                          'sampling_strategy': "not minority",
                          'replacement': "False",
                          'ab_n_estimators': 50,
                          'ab_learning_rate': 0.1,
                          'ab_algorithm': "SAMME.R",
                          'ab_max_depth': 1}
            return space
