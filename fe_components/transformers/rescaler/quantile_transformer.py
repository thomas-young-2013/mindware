from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    CategoricalHyperparameter
from fe_components.transformers.base_transformer import *


class QuantileTransformation(Transformer):
    def __init__(self, n_quantiles=1000, output_distribution='uniform', random_state=1):
        super().__init__("quantile_transformer", 5)
        self.input_type = [NUMERICAL, DISCRETE]
        self.compound_mode = 'in_place'
        self.output_type = NUMERICAL

        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.preprocessing import QuantileTransformer

        X, y = input_datanode.data
        X_new = X[:, target_fields]

        if not self.model:
            self.model = QuantileTransformer(output_distribution=self.output_distribution,
                                             n_quantiles=self.n_quantiles)
            self.model.fit(X_new)

        _X = self.model.transform(X_new)

        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        # TODO parametrize like the Random Forest as n_quantiles = n_features^param
        n_quantiles = UniformIntegerHyperparameter(
            'n_quantiles', lower=10, upper=2000, default_value=1000
        )
        output_distribution = CategoricalHyperparameter(
            'output_distribution', ['uniform', 'normal'], default_value="uniform"
        )
        cs.add_hyperparameters([n_quantiles, output_distribution])
        return cs
