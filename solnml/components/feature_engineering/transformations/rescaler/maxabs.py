from solnml.components.feature_engineering.transformations.base_transformer import *


class MaxabsScaler(Transformer):
    def __init__(self, **kwargs):
        super().__init__("maxabs_scaler", 42)
        self.input_type = [DISCRETE, NUMERICAL]
        self.compound_mode = 'in_place'
        self.output_type = NUMERICAL

    @ease_trans
    def operate(self, input_data, target_fields):
        from sklearn.preprocessing import MaxAbsScaler
        X, y = input_data.data
        X_new = X[:, target_fields]

        if not self.model:
            self.model = MaxAbsScaler()
            self.model.fit(X_new)

        _X = self.model.transform(X_new)

        return _X
