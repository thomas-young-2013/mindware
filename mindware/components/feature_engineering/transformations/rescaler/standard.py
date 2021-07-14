from mindware.components.feature_engineering.transformations.base_transformer import *


class StandardScaler(Transformer):
    type = 43

    def __init__(self, **kwargs):
        super().__init__("standard_scaler")
        self.input_type = [DISCRETE, NUMERICAL]
        self.compound_mode = 'in_place'
        self.output_type = NUMERICAL

    @ease_trans
    def operate(self, input_data, target_fields):
        from sklearn.preprocessing import StandardScaler
        X, y = input_data.data
        X_new = X[:, target_fields]

        if not self.model:
            self.model = StandardScaler()
            self.model.fit(X_new)

        _X = self.model.transform(X_new)

        return _X
