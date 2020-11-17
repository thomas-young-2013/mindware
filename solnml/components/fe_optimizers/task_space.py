from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from solnml.components.feature_engineering.transformations import _bal_balancer, _preprocessor, _rescaler, \
    _image_preprocessor, _text_preprocessor
from solnml.components.utils.constants import CLS_TASKS
from solnml.components.feature_engineering import TRANS_CANDIDATES


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
        _trans_types = TRANS_CANDIDATES['classification'].copy()
        trans_types = TRANS_CANDIDATES['classification'].copy()
    else:
        _trans_types = TRANS_CANDIDATES['regression'].copy()
        trans_types = TRANS_CANDIDATES['regression'].copy()

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
            if key not in _preprocessor:
                raise ValueError("Preprocessor %s not in built-in preprocessors!" % key)
            else:
                preprocessor[key] = _preprocessor[key]
        trans_types = _trans_types
    else:
        preprocessor = _preprocessor

    configs = dict()

    if include_image:
        image_preprocessor_dict = _get_configuration_space(_image_preprocessor, optimizer=optimizer)
        configs['image_preprocessor'] = image_preprocessor_dict
    if include_text:
        text_preprocessor_dict = _get_configuration_space(_text_preprocessor, optimizer=optimizer)
        configs['text_preprocessor'] = text_preprocessor_dict

    preprocessor_dict = _get_configuration_space(preprocessor, trans_types, optimizer=optimizer)
    configs['preprocessor'] = preprocessor_dict
    rescaler_dict = _get_configuration_space(_rescaler, trans_types, optimizer=optimizer)
    configs['rescaler'] = rescaler_dict
    if task_type in CLS_TASKS:
        _balancer = _bal_balancer
        balancer_dict = _get_configuration_space(_balancer, optimizer=optimizer)
    else:
        balancer_dict = None
    configs['balancer'] = balancer_dict
    cs = _build_hierachical_configspace(configs, optimizer=optimizer)
    return cs


def _get_configuration_space(builtin_transformers, trans_type=None, optimizer='smac'):
    config_dict = dict()
    for tran_key in builtin_transformers:
        tran = builtin_transformers[tran_key]
        tran_id = tran().type
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
                                              default_value=config_cand[-1])
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
