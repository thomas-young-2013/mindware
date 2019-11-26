from fe_components.transformers.base_transformer import *


class KernelPCA(Transformer):
    def __init__(self, param=100):
        super().__init__("kernel_pca", 12)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = NUMERICAL
        self.params = {'n_components': param}

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.decomposition import KernelPCA

        X, y = input_datanode.data

        if self.model is None:
            self.model = KernelPCA(n_components=self.params['n_components'], kernel='rbf')
            self.model.fit(X)
        X_new = self.model.transform(X)

        return X_new
