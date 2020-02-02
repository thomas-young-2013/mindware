from automlToolkit.components.feature_engineering.transformations.base_transformer import *


@DeprecationWarning
class CrossFeatureTransformation(Transformer):
    def __init__(self, random_state=None):
        super().__init__("cross_features", 17)
        self.input_type = [CATEGORICAL]
        self.compound_mode = 'concatenate'

        self.output_type = CATEGORICAL
        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields):
        from sklearn.preprocessing import PolynomialFeatures

        X, y = input_datanode.data
        X_new = X[:, target_fields]
        ori_length = X_new.shape[1]

        if not self.model:
            self.model = PolynomialFeatures(
                degree=2, interaction_only=True,
                include_bias=False)
            self.model.fit(X_new)

        _X = self.model.transform(X_new)
        if ori_length == 1:
            _X = _X[:, 1:]
        else:
            _X = _X[:, ori_length + 1:]

        return _X
