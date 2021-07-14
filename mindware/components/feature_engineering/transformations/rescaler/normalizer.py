from mindware.components.feature_engineering.transformations.base_transformer import *


class NormalizeTransformation(Transformer):
    type = 4

    def __init__(self):
        super().__init__("normalizer")
        self.input_type = [NUMERICAL, DISCRETE]
        self.compound_mode = 'in_place'
        self.params = {'norm': 'l2'}
        self.output_type = NUMERICAL

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.preprocessing import Normalizer

        X, y = input_datanode.data
        X_new = X[:, target_fields]

        if not self.model:
            self.model = Normalizer(norm=self.params['norm'])
            self.model.fit(X_new)

        _X = self.model.transform(X_new)

        return _X
