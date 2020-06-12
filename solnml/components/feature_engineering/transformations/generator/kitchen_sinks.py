import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from solnml.components.feature_engineering.transformations.base_transformer import *


class KitchenSinks(Transformer):
    def __init__(self, gamma=1.0, n_components=100, random_state=1):
        super().__init__("kitchen_sinks", 13, random_state=random_state)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data
        X_new = X[:, target_fields]

        if not self.model:
            import sklearn.kernel_approximation
            self.model = sklearn.kernel_approximation.RBFSampler(
                gamma=self.gamma, n_components=self.n_components, random_state=self.random_state)
            self.model.fit(X_new)

        _X = self.model.transform(X_new)

        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            gamma = UniformFloatHyperparameter(
                "gamma", 3.0517578125e-05, 8, default_value=1.0, log=True)
            n_components = UniformIntegerHyperparameter(
                "n_components", 50, 2000, default_value=100, log=True)
            cs = ConfigurationSpace()
            cs.add_hyperparameters([gamma, n_components])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'gamma': hp.loguniform('kitchen_gamma', np.log(3.05e05), np.log(8)),
                     'n_components': hp.randint('kitchen_n_components', 1950) + 50}
            return space
