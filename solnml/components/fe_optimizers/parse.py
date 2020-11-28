from solnml.components.feature_engineering.transformations import _bal_balancer, _preprocessor, _rescaler, \
    _image_preprocessor, _text_preprocessor, _bal_addons, _gen_addons, _res_addons, _sel_addons
from solnml.components.utils.class_loader import get_combined_fe_candidtates
from solnml.components.feature_engineering.transformation_graph import DataNode


def parse_config(data_node: DataNode, config, record=False, skip_balance=False):
    """
        Transform the data node based on the pipeline specified by configuration.
    :param data_node:
    :param config:
    :param record:
    :return: the resulting data node.
    """
    _preprocessor_candidates = get_combined_fe_candidtates(_preprocessor, _gen_addons)
    _preprocessor_candidates = get_combined_fe_candidtates(_preprocessor_candidates, _sel_addons)
    _rescaler_candidates = get_combined_fe_candidtates(_rescaler, _res_addons)
    _balancer_candadates = get_combined_fe_candidtates(_bal_balancer, _bal_addons)

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

    image_pre_id = config_dict.get('image_preprocessor', None)
    if image_pre_id:
        config_dict.pop('image_preprocessor')
    text_pre_id = config_dict.get('text_preprocessor', None)
    if text_pre_id:
        config_dict.pop('text_preprocessor')

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
    tran_dict = dict()

    # Image preprocessor
    if image_pre_id:
        _node, image_tran = tran_operate(image_pre_id, _image_preprocessor, config_dict, _node)
        tran_dict['image_preprocessor'] = image_tran

    # Text preprocessor
    if text_pre_id:
        _node, text_tran = tran_operate(text_pre_id, _text_preprocessor, config_dict, _node)
        tran_dict['text_preprocessor'] = text_tran

    # Balancer
    _balancer = _bal_balancer
    _node, bal_tran = tran_operate(bal_id, _balancer_candadates, config_dict, _node)
    tran_dict['balancer'] = bal_tran

    # Rescaler
    _node, res_tran = tran_operate(res_id, _rescaler_candidates, config_dict, _node)
    tran_dict['rescaler'] = res_tran

    # Generator
    _node, gen_tran = tran_operate(gen_id, _preprocessor_candidates, config_dict, _node)
    tran_dict['preprocessor'] = gen_tran

    _node.config = config
    if record:
        return _node, tran_dict
    return _node


def construct_node(data_node: DataNode, tran_dict, mode='test'):
    if 'image_preprocessor' in tran_dict:
        data_node = tran_dict['image_preprocessor'].operate(data_node)

    if 'text_preprocessor' in tran_dict:
        data_node = tran_dict['text_preprocessor'].operate(data_node)

    if mode != 'test':
        data_node = tran_dict['balancer'].operate(data_node)

    data_node = tran_dict['rescaler'].operate(data_node)
    data_node = tran_dict['preprocessor'].operate(data_node)
    return data_node
