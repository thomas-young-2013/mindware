from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from mindware.components.feature_engineering.transformations.base_transformer import *
from mindware.components.utils.operations import *


class BinaryTransformation(Transformer):
    type = 22

    def __init__(self, func='add'):
        super().__init__("binary_transformer")
        self.input_type = NUMERICAL
        self.output_type = NUMERICAL
        self.compound_mode = 'concatenate'
        self.func = func
        self.optional_params = ['add', 'sub', 'mul', 'div']

    # TODO: Unimplemented transformations.
    @ease_trans
    def operate(self, input_datanode: DataNode, target_fields=None):
        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(input_datanode.feature_types, self.input_type)
        X1 = X[:, target_fields]
        X2 = X[:, target_fields]

        if not self.model:
            self.get_model(self.func)
            self.model.fit(X1, X2)

        _X = self.model.transform(X1, X2)

        return _X

    def get_model(self, param):
        if param == 'add':
            self.model = Addition()
        elif param == 'sub':
            self.model = Subtract()
        elif param == 'mul':
            self.model = Multiply()
        elif param == 'div':
            self.model = Division()
        else:
            raise ValueError("Unknown param name %s!" % str(param))

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        optional_funcs = ['add', 'sub', 'mul', 'div']
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            func = CategoricalHyperparameter('func', optional_funcs, default_value='mul')
            cs.add_hyperparameter(func)
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'func': hp.choice('binary_func', optional_funcs)}
            return space
