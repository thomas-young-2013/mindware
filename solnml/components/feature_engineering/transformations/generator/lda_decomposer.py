from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
from solnml.components.feature_engineering.transformations.base_transformer import *
from solnml.components.utils.configspace_utils import check_none


@DeprecationWarning
class LdaDecomposer(Transformer):
    type = 14

    def __init__(self, shrinkage="None", n_components=None):
        super().__init__("lda_decomposer")
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = NUMERICAL
        self.compound_mode = 'only_new'
        self.shrinkage = shrinkage
        self.n_components = n_components

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        X, y = input_datanode.data

        if self.model is None:
            if check_none(self.shrinkage):
                self.shrinkage = None

            self.model = LinearDiscriminantAnalysis(
                n_components=self.n_components,
                shrinkage=self.shrinkage
            )
            self.model.fit(X, y)
        X_new = self.model.transform(X)

        return X_new

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        cs = ConfigurationSpace()
        shrinkage = CategoricalHyperparameter(
            "shrinkage", ["None", "auto"], default_value="None")
        # n_components = UniformIntegerHyperparameter('n_components', 1, 250, default_value=10)
        # cs.add_hyperparameters([shrinkage, n_components])
        cs.add_hyperparameter(shrinkage)
        return cs
