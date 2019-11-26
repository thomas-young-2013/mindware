from fe_components.transformers.base_transformer import *


class SvdDecomposer(Transformer):
    def __init__(self, frac=0.3):
        super().__init__("svd", 19)
        self.params = {'target_dim': frac}
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = NUMERICAL

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.decomposition import TruncatedSVD

        X, y = input_datanode.data

        if self.model is None:
            n_components = int(X.shape[1] * self.params['target_dim'])
            self.model = TruncatedSVD(n_components=n_components, algorithm='randomized')
            self.model.fit(X)
        X_new = self.model.transform(X)

        return X_new
