from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from automlToolkit.components.feature_engineering.transformations.base_transformer import Transformer, ease_trans
from automlToolkit.components.utils.constants import *
from automlToolkit.components.utils.operations import *


class ArithmeticTransformation(Transformer):
    def __init__(self, func='sqrt'):
        super().__init__("arithmetic_transformer", 21)
        self.input_type = [NUMERICAL, DISCRETE]
        self.output_type = NUMERICAL
        self.compound_mode = 'in_place'
        self.func = func

    @ease_trans
    def operate(self, input_datanode, target_fields):
        X, y = input_datanode.data
        X_new = X[:, target_fields]

        X_new = np.array(X_new.tolist())
        if not self.model:
            self.get_model(self.func)
            self.model.fit(X_new)

        _X = self.model.transform(X_new)
        return _X

    def get_model(self, param):
        if param == 'log':
            self.model = Log()
        elif param == 'sqrt':
            self.model = Sqrt()
        elif param == 'square':
            self.model = Square()
        elif param == 'freq':
            self.model = Freq()
        elif param == 'round':
            self.model = Round()
        elif param == 'sigmoid':
            self.model = Sigmoid()
        elif param == 'tanh':
            self.model = Tanh()
        else:
            raise ValueError("Unknown param name %s!" % str(param))

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        optional_funcs = ['log', 'sqrt', 'square', 'freq', 'round', 'sigmoid', 'tanh']
        scaler = CategoricalHyperparameter('func', optional_funcs, default_value='sqrt')
        cs.add_hyperparameter(scaler)
        return cs
