from solnml.components.feature_engineering.transformations import _imb_balancer, _bal_balancer, _preprocessor, \
    _rescaler
from solnml.components.feature_engineering.transformation_graph import DataNode


def parse_config(data_node: DataNode, config, record=False, skip_balance=False):
    """
        Transform the data node based on the pipeline specified by configuration.
    :param data_node:
    :param config:
    :param record:
    :return: the resulting data node.
    """
    # Remove the indicator in config_dict.
    config_dict = config.get_dictionary().copy()

    if skip_balance:
        bal_id = 'empty'
    else:
        if 'balancer' in config_dict:
            bal_id = config_dict['balancer']
            config_dict.pop('balancer')
        else:
            bal_id = 'empty'
    gen_id = config_dict['preprocessor']
    config_dict.pop('preprocessor')
    res_id = config_dict['rescaler']
    config_dict.pop('rescaler')

    def tran_operate(id, tran_set, config, node):
        _config = {}
        for key in config:
            if id in key:
                config_name = key.split(':')[1]
                _config[config_name] = config[key]
        tran = tran_set[id](**_config)
        output_node = tran.operate(node)
        return output_node, tran

    _node = data_node.copy_()

    # Balancer
    _balancer = _bal_balancer
    _node, bal_tran = tran_operate(bal_id, _balancer, config_dict, _node)

    # Rescaler
    _node, res_tran = tran_operate(res_id, _rescaler, config_dict, _node)

    # Generator
    _node, gen_tran = tran_operate(gen_id, _preprocessor, config_dict, _node)

    _node.config = config
    if record:
        return _node, [bal_tran, res_tran, gen_tran]
    return _node


def construct_node(data_node: DataNode, op_list, mode='test'):
    if mode != 'test':
        if op_list[0] is not None:
            data_node = op_list[0].operate(data_node)
    if op_list[1] is not None:
        data_node = op_list[1].operate(data_node)
    if op_list[2] is not None:
        data_node = op_list[2].operate(data_node)
    return data_node
