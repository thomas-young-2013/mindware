from fe_components.transformers.base_transformer import *


class PcaDecomposer(Transformer):
    def __init__(self, frac=0.618):
        super().__init__("pca_decomposer", 6)
        self.params = {'frac': frac}
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = NUMERICAL

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.decomposition import PCA

        X, y = input_datanode.data

        if self.model is None:
            self.model = PCA(n_components=self.params['frac'],
                             whiten=False)
            self.model.fit(X)
        X_new = self.model.transform(X)

        return X_new
