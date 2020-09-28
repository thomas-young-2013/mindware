from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from solnml.components.feature_engineering.transformations.base_transformer import *


class RobustScaler(Transformer):
    def __init__(self, q_min=0.25, q_max=0.75, **kwargs):
        super().__init__("robust_scaler", 44)
        self.input_type = [DISCRETE, NUMERICAL]
        self.compound_mode = 'in_place'
        self.output_type = NUMERICAL

        self.q_min = q_min
        self.q_max = q_max

    @ease_trans
    def operate(self, input_data, target_fields):
        from sklearn.preprocessing import RobustScaler
        X, y = input_data.data
        X_new = X[:, target_fields]

        if not self.model:
            self.model = RobustScaler(quantile_range=(self.q_min, self.q_max))
            self.model.fit(X_new)

        _X = self.model.transform(X_new)

        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        cs = ConfigurationSpace()
        q_min = UniformFloatHyperparameter(
            'q_min', 0.001, 0.3, default_value=0.25
        )
        q_max = UniformFloatHyperparameter(
            'q_max', 0.7, 0.999, default_value=0.75
        )
        cs.add_hyperparameters((q_min, q_max))
        return cs
