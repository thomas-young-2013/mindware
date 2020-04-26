from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from automlToolkit.components.feature_engineering.transformations.base_transformer import *


class KBinsDiscretizer(Transformer):
    def __init__(self, n_bins=3, strategy='uniform'):
        super().__init__("discretizer", 24)
        self.input_type = NUMERICAL
        self.output_type = DISCRETE
        self.compound_mode = 'in_place'
        self.n_bins = n_bins
        self.strategy = strategy

    @ease_trans
    def operate(self, input_datanode: DataNode, target_fields=None):
        from sklearn.preprocessing import KBinsDiscretizer

        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(input_datanode.feature_types, self.input_type)
        X_new = X[:, target_fields]

        if not self.model:
            self.model = KBinsDiscretizer(
                n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
            self.model.fit(X_new)
        _X = self.model.transform(X_new)
        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_bins = UniformIntegerHyperparameter('n_bins', 2, 20, default_value=5)
        cs.add_hyperparameters([n_bins])
        return cs
