from fe_components.transformers.base_transformer import *


class KBinsDiscretizeTransformation(Transformer):
    def __init__(self, n_bins=50):
        super().__init__("discretizer", 3)
        self.input_type = NUMERICAL
        self.output_type = DISCRETE
        self.model = None
        self.params = {'n_bins': n_bins}

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.preprocessing import KBinsDiscretizer

        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(input_datanode.feature_types, self.input_type)
        X_new = X[:, target_fields]

        if not self.model:
            self.model = KBinsDiscretizer(n_bins=self.params['n_bins'], encode='ordinal')
            self.model.fit(X_new)
        _X = self.model.transform(X_new)

        return _X
