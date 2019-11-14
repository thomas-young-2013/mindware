from fe_components.transformers.base_transformer import *


class RandomTreesEmbeddingTransformation(Transformer):
    def __init__(self, distribution='uniform'):
        super().__init__("random_trees_embedding", 7)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = CATEGORICAL
        self.params = distribution

    @ease_trans
    def operate(self, input_datanode: DataNode, target_fields=None):
        from sklearn.ensemble import RandomTreesEmbedding

        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(input_datanode.feature_types, self.input_type)
        X_new = X[:, target_fields]

        if not self.model:
            self.model = RandomTreesEmbedding()
            self.model.fit(X_new)

        _X = self.model.transform(X_new).toarray()

        return _X
