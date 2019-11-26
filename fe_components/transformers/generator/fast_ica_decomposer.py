from fe_components.transformers.base_transformer import *


class FastIcaDecomposer(Transformer):
    def __init__(self, frac=0.3):
        super().__init__("fast_ica", 10)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = NUMERICAL
        self.params = frac

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.decomposition import FastICA

        X, y = input_datanode.data

        if self.model is None:
            self.model = FastICA(n_components=int(X.shape[1] * self.params))
            self.model.fit(X)
        X_new = self.model.transform(X)

        return X_new
