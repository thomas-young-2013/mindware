from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from solnml.components.feature_engineering.transformations.base_transformer import *


class SvdDecomposer(Transformer):
    def __init__(self, target_dim=128, random_state=1):
        super().__init__("svd", 19)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.target_dim = target_dim
        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data

        if self.model is None:
            from sklearn.decomposition import TruncatedSVD

            self.target_dim = int(self.target_dim)
            target_dim = min(self.target_dim, X.shape[1] - 1)
            self.model = TruncatedSVD(
                target_dim, algorithm='randomized')
            # TODO: remove when migrating to sklearn 0.16
            # Circumvents a bug in sklearn
            # https://github.com/scikit-learn/scikit-learn/commit/f08b8c8e52663167819f242f605db39f3b5a6d0c
            # X = X.astype(np.float64)
            self.model.fit(X, y)

        X_new = self.model.transform(X)

        return X_new

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        target_dim = UniformIntegerHyperparameter(
            "target_dim", 10, 256, default_value=128)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(target_dim)
        return cs
