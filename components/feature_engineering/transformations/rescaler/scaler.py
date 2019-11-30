from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from components.feature_engineering.transformations.base_transformer import *


class ScaleTransformation(Transformer):
    def __init__(self, scaler='min_max'):
        super().__init__("scaler", 3)
        self.input_type = [DISCRETE, NUMERICAL]
        self.compound_mode = 'in_place'
        self.output_type = NUMERICAL
        self.scaler = scaler

    def get_model(self, param):
        from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

        if param == 'min_max':
            return MinMaxScaler()
        elif param == 'max_abs':
            return MaxAbsScaler()
        elif param == 'standard':
            return StandardScaler()
        elif param == 'robust':
            return RobustScaler()
        else:
            raise ValueError('Invalid param!')

    @ease_trans
    def operate(self, input_data, target_fields):
        X, y = input_data.data
        X_new = X[:, target_fields]

        if not self.model:
            self.model = self.get_model(self.scaler)
            self.model.fit(X_new)
        _X = self.model.transform(X_new)

        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        scaler = CategoricalHyperparameter(
            'scaler', ['min_max', 'max_abs', 'standard', 'robust'], default_value='min_max'
        )
        cs.add_hyperparameter(scaler)
        return cs
