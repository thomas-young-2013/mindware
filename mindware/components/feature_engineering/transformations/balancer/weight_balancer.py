from mindware.components.feature_engineering.transformations.base_transformer import *


class WeightBalancer(Transformer):
    type = 20

    def __init__(self, random_state=1):
        super().__init__("weight_balancer")
        self.random_state = random_state

    def operate(self, input_datanode: DataNode, target_fields=None):
        output_datanode = input_datanode.copy_()
        if output_datanode.data_balance != 1:
            output_datanode.enable_balance = 1
        output_datanode.trans_hist.append(self.type)
        return output_datanode
