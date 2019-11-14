from fe_components.transformers.base_transformer import *


class SvdDecomposer(Transformer):
    def __init__(self, frac=0.3):
        super().__init__("svd_decomposer", 6)
        self.params = frac
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = NUMERICAL

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.decomposition import TruncatedSVD

        X, y = input_datanode.data

        if self.model is None:
            self.model = TruncatedSVD(n_components=int(X.shape[1] * self.params), n_iter=7)
            self.model.fit(X)
        X_new = self.model.transform(X)

        return X_new
