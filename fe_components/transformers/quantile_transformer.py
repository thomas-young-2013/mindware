from fe_components.transformers.base_transformer import *


class QuantileTransformation(Transformer):
    def __init__(self, distribution='uniform'):
        super().__init__("quantile_transformer", 4)
        self.input_type = [NUMERICAL, DISCRETE]
        self.output_type = NUMERICAL
        self.params = {'distribution': distribution}

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.preprocessing import QuantileTransformer

        X, y = input_datanode.data
        X_new = X[:, target_fields]

        if not self.model:
            self.model = QuantileTransformer(output_distribution=self.params['distribution'])
            self.model.fit(X_new)

        _X = self.model.transform(X_new)

        return _X
