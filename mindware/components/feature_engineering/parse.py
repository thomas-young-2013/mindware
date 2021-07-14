from mindware.components.feature_engineering.transformations import _bal_balancer, _preprocessor, _rescaler, \
    _image_preprocessor, _text_preprocessor, _bal_addons, _imb_balancer, _gen_addons, _res_addons, _sel_addons
from mindware.components.utils.class_loader import get_combined_fe_candidtates
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.feature_engineering.task_space import stage_list, thirdparty_candidates_dict


def parse_config(data_node: DataNode, config: dict, record=False, skip_balance=False, if_imbal=False):
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

    if not if_imbal:
        _balancer_candidates = get_combined_fe_candidtates(_bal_balancer, _bal_addons)
    else:
        _balancer_candidates = get_combined_fe_candidtates(_imb_balancer, _bal_addons)

    # Remove the indicator in config_dict.
    config_dict = config.copy()

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

    for stage in stage_list:
        if stage == 'balancer':
            if skip_balance:
                op_id = 'empty'
            else:
                if stage in config_dict:
                    op_id = config_dict[stage]
                    config_dict.pop(stage)
                else:
                    op_id = 'empty'
        else:
            op_id = config_dict[stage]
            config_dict.pop(stage)
        if stage == 'preprocessor':
            _node, tran = tran_operate(op_id, _preprocessor_candidates, config_dict, _node)
        elif stage == 'rescaler':
            _node, tran = tran_operate(op_id, _rescaler_candidates, config_dict, _node)
        elif stage == 'balancer':
            _node, tran = tran_operate(op_id, _balancer_candidates, config_dict, _node)
        else:
            # Third party stage
            _node, tran = tran_operate(op_id, thirdparty_candidates_dict[stage], config_dict, _node)

        tran_dict[stage] = tran

    _node.config = config
    if record:
        return _node, tran_dict
    return _node


def construct_node(data_node: DataNode, tran_dict, mode='test'):
    if 'image_preprocessor' in tran_dict:
        data_node = tran_dict['image_preprocessor'].operate(data_node)

    if 'text_preprocessor' in tran_dict:
        data_node = tran_dict['text_preprocessor'].operate(data_node)

    for stage in stage_list:
        if stage_list == 'balancer' and mode == 'test':
            continue
        data_node = tran_dict[stage].operate(data_node)
    return data_node
