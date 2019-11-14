from fe_components.transformers.base_transformer import *


class ScaleTransformation(Transformer):
    def __init__(self, param='min_max'):
        super().__init__("scaler", 1)
        self.input_type = [DISCRETE, NUMERICAL]
        self.output_type = NUMERICAL
        self.params = {'func': param}
        self.optional_params = ['min_max', 'max_abs', 'standard', 'robust']

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
            self.model = self.get_model(self.params['func'])
            self.model.fit(X_new)
        _X = self.model.transform(X_new)

        return _X
