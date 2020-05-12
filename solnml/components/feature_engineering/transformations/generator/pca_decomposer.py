from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter
from solnml.components.feature_engineering.transformations.base_transformer import *
from solnml.components.utils.configspace_utils import check_for_bool


class PcaDecomposer(Transformer):
    def __init__(self, keep_variance=0.9999, whiten='False', random_state=1):
        super().__init__("pca", 16)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.keep_variance = keep_variance
        self.whiten = whiten
        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data

        if self.model is None:
            import sklearn.decomposition
            n_components = float(self.keep_variance)
            self.whiten = check_for_bool(self.whiten)
            self.model = sklearn.decomposition.PCA(n_components=n_components,
                                                   whiten=self.whiten,
                                                   copy=True)
            self.model.fit(X)

            if not np.isfinite(self.model.components_).all():
                raise ValueError("PCA found non-finite components.")

        X_new = self.model.transform(X)

        return X_new

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        keep_variance = UniformFloatHyperparameter(
            "keep_variance", 0.5, 0.9999, default_value=0.9999)
        whiten = CategoricalHyperparameter(
            "whiten", ["False", "True"], default_value="False")
        cs = ConfigurationSpace()
        cs.add_hyperparameters([keep_variance, whiten])
        return cs
