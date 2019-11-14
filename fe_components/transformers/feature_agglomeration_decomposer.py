from fe_components.transformers.base_transformer import *


class FeatureAgglomerationDecomposer(Transformer):
    def __init__(self, frac=0.3):
        super().__init__("feature_agglomeration_decomposer", 6)
        self.params = frac
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = NUMERICAL

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.cluster import FeatureAgglomeration

        X, y = input_datanode.data

        if self.model is None:
            self.model = FeatureAgglomeration(n_clusters=int(X.shape[1] * self.params))
            self.model.fit(X)
        X_new = self.model.transform(X)

        return X_new
