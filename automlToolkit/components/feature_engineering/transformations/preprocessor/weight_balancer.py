from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from automlToolkit.components.feature_engineering.transformations.base_transformer import *


class WeightBalancer(Transformer):
    def __init__(self, balance_type=1, random_state=1):
        super().__init__("weight_balancer", 20)
        self.balance_type = balance_type
        self.random_state = random_state

    def operate(self, input_datanode: DataNode, target_fields=None):
        output_datanode = input_datanode.copy_()
        output_datanode.enable_balance = self.balance_type
        output_datanode.trans_hist.append(self.type)
        return output_datanode

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        balance_type = CategoricalHyperparameter(
            'balance_type', [0, 1], default_value=1)
        cs.add_hyperparameter(balance_type)
        return cs
