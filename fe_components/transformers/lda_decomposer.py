from fe_components.transformers.base_transformer import *


class LdaDecomposer(Transformer):
    def __init__(self, frac=0.3):
        super().__init__("lda_decomposer", 6)
        self.params = frac
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = NUMERICAL

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        X, y = input_datanode.data

        if self.model is None:
            self.model = LinearDiscriminantAnalysis(n_components=int(X.shape[1] * self.params))
            self.model.fit(X, y)
        X_new = self.model.transform(X)

        return X_new
