from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    Constant, UniformIntegerHyperparameter

from automlToolkit.components.models.base_model import BaseRegressionModel
from automlToolkit.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class KNearestNeighborsRegressor(BaseRegressionModel):

    def __init__(self, n_neighbors, weights, p, random_state=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state
        self.n_jobs = 1
        self.time_limit = None

    def fit(self, X, Y):
        from sklearn.neighbors import KNeighborsRegressor
        self.estimator = KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                             weights=self.weights,
                                             p=self.p)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KNN',
                'name': 'K-Nearest Neighbor Regression',
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

            n_neighbors = UniformIntegerHyperparameter(
                name="n_neighbors", lower=1, upper=100, log=True, default_value=1)
            weights = CategoricalHyperparameter(
                name="weights", choices=["uniform", "distance"], default_value="uniform")
            p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
            cs.add_hyperparameters([n_neighbors, weights, p])

            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_neighbors': hp.randint('knn_n_neighbors', 100) + 1,
                     'weights': hp.choice('knn_weights', ['uniform', 'distance']),
                     'p': hp.choice('knn_p', [1, 2])}

            init_trial = {'n_neighbors': 1, 'weights': "uniform", 'p': 2}

            return space
