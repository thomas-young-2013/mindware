import warnings
from collections import OrderedDict
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from solnml.components.feature_engineering.transformations import _bal_balancer, _preprocessor, _rescaler, \
    _image_preprocessor, _text_preprocessor, _bal_addons, _gen_addons, _res_addons, _sel_addons, EmptyTransformer
from solnml.components.utils.class_loader import get_combined_fe_candidtates
from solnml.components.utils.constants import CLS_TASKS
from solnml.components.feature_engineering import TRANS_CANDIDATES

builtin_stage = ['balancer', 'preprocessor', 'rescaler']
stage_list = ['balancer', 'preprocessor', 'rescaler']
thirdparty_candidates_dict = OrderedDict()


def set_stage(udf_stage_list, stage_candidates_dict):
    '''
    :param udf_stage_list: List, a list for stage_name like ['my_stage','selector']
    :param stage_candidates_dict: Dictionary, <key, value>.
        Key is stage_name, and value is a list of operators in this stage.
        Each operator must be a Transformer.
    :return:
    '''
    global stage_list
    stage_list = udf_stage_list
    print("Current Stage: %s" % ', '.join(stage_list))
    for stage in udf_stage_list:
        if stage in builtin_stage:
            print("Built-in stage '%s' found!" % stage)
        else:
            print("User-defined stage '%s' found!" % stage)
            if stage not in stage_candidates_dict:
                raise ValueError("Expected stage name '%s' in stage_candidates_dict." % stage)
            if len(stage_candidates_dict[stage]) == 0:
                warnings.warn("Candidate list for stage '%s' is empty! EmptyTransformer will be used instead！" % stage)
                stage_candidates_dict[stage] = [EmptyTransformer]
            thirdparty_candidates_dict[stage] = {candidate.__name__: candidate for candidate in
                                                 stage_candidates_dict[stage]}


def get_task_hyperparameter_space(task_type, estimator_id, include_preprocessors=None,
                                  include_text=False, include_image=False,
                                  optimizer='smac'):
    """
        Fetch the underlying hyperparameter space for feature engineering.
        Pipeline Space:
            1. balancer: weight_balancer,
                         data_balancer.
            2. scaler: normalizer, scaler, quantile.
            3. preprocessor
    :return: hyper space.
    """
    if task_type in CLS_TASKS:
        trans_types = TRANS_CANDIDATES['classification'].copy()
    else:
        trans_types = TRANS_CANDIDATES['regression'].copy()

    _preprocessor_candidates, trans_types = get_combined_fe_candidtates(_preprocessor, _gen_addons, trans_types)
    _preprocessor_candidates, trans_types = get_combined_fe_candidtates(_preprocessor_candidates, _sel_addons,
                                                                        trans_types)
    _rescaler_candidates, trans_types = get_combined_fe_candidtates(_rescaler, _res_addons, trans_types)
    _balancer_candadates, trans_types = get_combined_fe_candidtates(_bal_balancer, _bal_addons, trans_types)

    # Avoid transformations, which would take too long
    # Combinations of non-linear models with feature learning.
    # feature_learning = ["kitchen_sinks", "kernel_pca", "nystroem_sampler"]
    if task_type in CLS_TASKS:
        classifier_set = ["adaboost", "decision_tree", "extra_trees",
                          "gradient_boosting", "k_nearest_neighbors",
                          "libsvm_svc", "random_forest", "gaussian_nb",
                          "decision_tree", "lightgbm"]

        if estimator_id in classifier_set:
            for tran_id in [12, 13, 15]:
                if tran_id in trans_types:
                    trans_types.remove(tran_id)

    preprocessor = dict()
    if include_preprocessors:
        for key in include_preprocessors:
            if key not in _preprocessor_candidates:
                raise ValueError(
                    "Preprocessor %s not in built-in preprocessors! Only the following preprocessors are supported: %s." % (
                        key, ','.join(_preprocessor_candidates.keys())))

            preprocessor[key] = _preprocessor_candidates[key]
            trans_types.append(_preprocessor_candidates[key].type)
    else:
        preprocessor = _preprocessor_candidates

    configs = dict()

    if include_image:
        image_preprocessor_dict = _get_configuration_space(_image_preprocessor, optimizer=optimizer)
        configs['image_preprocessor'] = image_preprocessor_dict
    if include_text:
        text_preprocessor_dict = _get_configuration_space(_text_preprocessor, optimizer=optimizer)
        configs['text_preprocessor'] = text_preprocessor_dict

    for stage in stage_list:
        if stage == 'preprocessor':
            stage_dict = _get_configuration_space(preprocessor, trans_types, optimizer=optimizer)
        elif stage == 'rescaler':
            stage_dict = _get_configuration_space(_rescaler_candidates, trans_types, optimizer=optimizer)
        elif stage == 'balancer':
            if task_type in CLS_TASKS:
                _balancer = _balancer_candadates
                stage_dict = _get_configuration_space(_balancer, optimizer=optimizer)
            else:
                stage_dict = None
        else:
            # Third party stage
            trans_types.extend([candidate.type for _, candidate in thirdparty_candidates_dict[stage].items()])
            stage_dict = _get_configuration_space(thirdparty_candidates_dict[stage], trans_types, optimizer=optimizer)
        configs[stage] = stage_dict

    cs = _build_hierachical_configspace(configs, optimizer=optimizer)
    return cs


def _get_configuration_space(builtin_transformers, trans_type=None, optimizer='smac'):
    config_dict = dict()
    for tran_key in builtin_transformers:
        tran = builtin_transformers[tran_key]
        tran_id = tran.type
        if trans_type is None or tran_id in trans_type:
            try:
                sub_configuration_space = builtin_transformers[tran_key].get_hyperparameter_search_space(
                    optimizer=optimizer)
                config_dict[tran_key] = sub_configuration_space
            except:
                if optimizer == 'smac':
                    config_dict[tran_key] = ConfigurationSpace()
                elif optimizer == 'tpe':
                    config_dict[tran_key] = {}
    return config_dict


def _add_hierachical_configspace(cs, config, parent_name):
    config_cand = list(config.keys())
    config_option = CategoricalHyperparameter(parent_name, config_cand,
                                              default_value=config_cand[0])
    cs.add_hyperparameter(config_option)
    for config_item in config_cand:
        sub_configuration_space = config[config_item]
        parent_hyperparameter = {'parent': config_option,
                                 'value': config_item}
        cs.add_configuration_space(config_item, sub_configuration_space,
                                   parent_hyperparameter=parent_hyperparameter)


def _build_hierachical_configspace(configs, optimizer='smac'):
    if optimizer == 'smac':
        cs = ConfigurationSpace()
        for config_key in configs:
            if configs[config_key] is not None:
                _add_hierachical_configspace(cs, configs[config_key], config_key)
        return cs
    elif optimizer == 'tpe':
        from hyperopt import hp
        space = {}

        def dict2hi(dictionary):
            hi_list = list()
            for key in dictionary:
                hi_list.append((key, dictionary[key]))
            return hi_list

        for config_key in configs:
            if configs[config_key] is not None:
                space[config_key] = hp.choice(config_key, dict2hi(configs[config_key]))

        return space
