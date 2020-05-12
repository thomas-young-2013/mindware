from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from solnml.components.feature_engineering.transformations.base_transformer import *


class ScaleTransformation(Transformer):
    def __init__(self, scaler='min_max', **kwargs):
        super().__init__("scaler", 3)
        self.input_type = [DISCRETE, NUMERICAL]
        self.compound_mode = 'in_place'
        self.output_type = NUMERICAL
        self.scaler = scaler
        self.ignore_flag = False

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
        import numpy as np
        X, y = input_data.data
        X_new = X[:, target_fields]

        if not self.model:
            self.model = self.get_model(self.scaler)
            self.model.fit(X_new)

        if self.scaler == 'robust':
            _flag = False
            for _id in range(X_new.shape[1]):
                _data = X_new[:, _id]
                if (np.percentile(_data, 0.75) - np.percentile(_data, 0.25)) > 5.:
                    _flag = True
            if not _flag:
                self.ignore_flag = True

        if not self.ignore_flag:
            _X = self.model.transform(X_new)
        else:
            _X = X_new.copy()
        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        scaler = CategoricalHyperparameter(
            'scaler', ['min_max', 'max_abs', 'standard', 'robust'], default_value='min_max'
        )
        cs.add_hyperparameter(scaler)
        return cs
