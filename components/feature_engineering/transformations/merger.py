from components.feature_engineering.transformations.base_transformer import *


class Merger(Transformer):
    def __init__(self):
        super().__init__("merger", 26)

    def operate(self, input_datanodes, target_fields=None):
        if type(input_datanodes) is not list:
            return input_datanodes

        X, y = input_datanodes[0].data
        self.target_fields = target_fields

        new_X = X.copy()
        new_feature_types = input_datanodes[0].feature_types.copy()

        for data_node in input_datanodes[1:]:
            new_X = np.hstack((new_X, data_node.data[0]))
            new_feature_types.extend(data_node.feature_types)
        output_datanode = DataNode((new_X, y), new_feature_types, input_datanodes[0].task_type)

        return output_datanode
