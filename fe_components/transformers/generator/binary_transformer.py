from fe_components.transformers.base_transformer import *
from fe_components.utils.operations import *


class BinaryTransformation(Transformer):
    def __init__(self, param='add'):
        super().__init__("binary_transformer", 22)
        self.input_type = NUMERICAL
        self.output_type = NUMERICAL
        self.params = param
        self.optional_params = ['add', 'sub', 'mul', 'div']

    # Unsafe to use decorator.
    # The decorator receives only one input node. Errors will occur if concatenated or target_fields is given!!
    @ease_trans
    def operate(self, input_datanode_first, input_datanode_second, target_fields=None):
        X_first, y_first = input_datanode_first.data
        X_second, y_second = input_datanode_second.data
        if target_fields is None:
            target_fields = collect_fields(input_datanode_first.feature_types, self.input_type)
        X_new_first = X_first[:, target_fields]
        X_new_second = X_second[:, target_fields]

        if not self.model:
            self.get_model(self.params)
            self.model.fit(X_new_first, X_new_second)

        _X = self.model.transform(X_new_first, X_new_second)

        return _X

    def get_model(self, param):
        if param in ['add', 'addition']:
            self.model = Addition()
        elif param in ['sub', 'subtract']:
            self.model = Subtract()
        elif param in ['mul', 'multiply']:
            self.model = Multiply()
        elif param in ['div', 'division']:
            self.model = Division()
        else:
            raise ValueError("Unknown param name %s!" % str(param))
