from fe_components.transformers.base_transformer import *


class PolynomialTransformation(Transformer):
    def __init__(self, degree=2):
        super().__init__("polynomial_generator", 10)

        self.input_type = [DISCRETE, NUMERICAL]
        self.output_type = NUMERICAL
        self.params = degree

    @ease_trans
    def operate(self, input_datanode, target_fields):
        from sklearn.preprocessing import PolynomialFeatures

        X, y = input_datanode.data
        X_new = X[:, target_fields]
        ori_length = X_new.shape[1]

        if not self.model:
            self.model = PolynomialFeatures(degree=self.params, interaction_only=True)
            self.model.fit(X_new)

        _X = self.model.transform(X_new)
        if ori_length == 1:
            _X = _X[:, 1:]
        else:
            _X = _X[:, ori_length + 1:]

        return _X
