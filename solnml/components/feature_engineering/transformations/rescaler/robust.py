from solnml.components.feature_engineering.transformations.base_transformer import *


class RobustScaler(Transformer):
    def __init__(self, **kwargs):
        super().__init__("robust_scaler", 44)
        self.input_type = [DISCRETE, NUMERICAL]
        self.compound_mode = 'in_place'
        self.output_type = NUMERICAL
        self.ignore_flag = False

    @ease_trans
    def operate(self, input_data, target_fields):
        from sklearn.preprocessing import RobustScaler
        X, y = input_data.data
        X_new = X[:, target_fields]

        if not self.model:
            self.model = RobustScaler()
            self.model.fit(X_new)

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
