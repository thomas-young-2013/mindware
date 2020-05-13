from solnml.components.feature_engineering.transformation_graph import DataNode


def get_increasing_sequence(data):
    """
        Return the increasing sequence.
    :param data:
    :return:
    """
    increasing_sequence = [data[0]]
    for item in data[1:]:
        _inc = increasing_sequence[-1] if item <= increasing_sequence[-1] else item
        increasing_sequence.append(_inc)
    return increasing_sequence


def is_unbalanced_dataset(data_node: DataNode):
    """
        Identify this dataset is balanced or not.
    :param data_node:
    :return: boolean.
    """
    labels = list(data_node.data[1])
    cnts = list()
    for val in set(labels):
        cnts.append(labels.count(val))
    cnts = sorted(cnts)
    # print('label distribution', cnts)
    assert len(cnts) > 1
    return cnts[0] * 4 <= cnts[-1]
