import time

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    Constant
from hyperopt import hp

from automlToolkit.components.models.base_model import BaseClassificationModel, IterativeComponentWithSampleWeight
from automlToolkit.components.utils.configspace_utils import check_none, check_for_bool
from automlToolkit.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from automlToolkit.components.utils.model_util import convert_multioutput_multiclass_to_multilabel


class ExtraTreesClassifier(IterativeComponentWithSampleWeight, BaseClassificationModel):

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split, max_features, bootstrap, random_state=None):

        if check_none(n_estimators):
            self.n_estimators = None
        else:
            self.n_estimators = int(self.n_estimators)
        self.criterion = criterion

        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = -1
        self.random_state = random_state

        self.estimator = None
        self.start_time = time.time()
        self.time_limit = None

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import ExtraTreesClassifier
        self.bootstrap = check_for_bool(self.bootstrap)
        self.estimator = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                              max_leaf_nodes=None,
                                              criterion=self.criterion,
                                              max_features=self.max_features,
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf,
                                              max_depth=None,
                                              bootstrap=self.bootstrap,
                                              random_state=self.random_state,
                                              n_jobs=self.n_jobs)
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

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ET',
                'name': 'Extra Trees Classifier',
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

            bootstrap = CategoricalHyperparameter(
                "bootstrap", ["True", "False"], default_value="False")
            cs.add_hyperparameters([n_estimators, criterion, max_features, min_samples_split, min_samples_leaf,
                                    bootstrap])
            return cs
        elif optimizer == 'tpe':
            space = {'n_estimators': hp.choice('et_n_estimators', [100]),
                     'criterion': hp.choice('et_criterion', ["gini", "entropy"]),
                     'max_features': hp.uniform('et_max_features', 0, 1),
                     'min_samples_split': hp.randint('et_min_samples_split', 19) + 2,
                     'min_samples_leaf': hp.randint('et_min_samples_leaf,', 20) + 1,
                     'bootstrap': hp.choice('et_bootstrap', ["True", "False"])}

            init_trial = {'n_estimators': 100, 'criterion': "gini", 'max_features': 0.5,
                          'min_samples_split': 2, 'min_samples_leaf': 1, 'bootstrap': "False"}
            return space
