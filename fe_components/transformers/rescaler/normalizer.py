from fe_components.transformers.base_transformer import *


class NormalizeTransformation(Transformer):
    def __init__(self, norm='l2'):
        super().__init__("normalizer", 4)
        self.input_type = [NUMERICAL, DISCRETE]
        self.output_type = NUMERICAL
        self.params = {'norm': norm}

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
