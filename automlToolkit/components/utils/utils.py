import sys
import pkgutil
import inspect
import importlib
import numpy as np
from collections import OrderedDict


def collect_fields(feature_types, target_type):
    if not isinstance(target_type, list):
        target_type = [target_type]
    return [idx for idx, type in enumerate(feature_types) if type in target_type]


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                                obj != base_class:
                    # TODO test if the obj implements the interface
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    transformer = obj
                    components[module_name] = transformer

    return components


def collect_infos(transformer_dict, feature_types):
    type_infos = dict()
    params_infos = dict()

    transformer_list = transformer_dict.keys()
    for feature_type in feature_types:
        type_infos[feature_type] = list()
        for transformer_id in transformer_list:
            transformer = transformer_dict[transformer_id]()
            target_fields = transformer.input_type
            if target_fields is None:
                continue
            if not isinstance(target_fields, list):
                target_fields = [target_fields]
            if feature_type in target_fields:
                type_infos[feature_type].append(transformer_id)

    for transformer_id in transformer_list:
        params_infos[transformer_id] = list()
        transformer = transformer_dict[transformer_id]()
        optional_params = transformer.optional_params
        if optional_params is not None:
            params_infos[transformer_id].extend(optional_params)

    return type_infos, params_infos


def is_numeric(n):
    try:
        float(n)  # Type-casting the string to `float`.
        # If string is not a valid `float`,
        # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True


def is_discrete(values):
    try:
        col_float = values.astype(np.float64)
        col_int = values.astype(np.int32)
        if (col_float == col_int).all():
            return True
        else:
            return False
    except:
        return False


def detect_categorical_type(values, threshold=0.001):
    total_cnts = len(values)
    unique_cnts = len(set(values))
    return bool(unique_cnts/total_cnts < threshold or unique_cnts <= 10)


def detect_abnormal_type(column_values):
    total_cnts = len(column_values)
    numeric_cnts, str_cnts = 0, 0
    str_idx, numeric_idx = list(), list()

    for idx, val in enumerate(column_values):
        try:
            val = str(val)
            if val == 'nan':
                total_cnts -= 1
                continue
            if is_numeric(val):
                numeric_cnts += 1
                numeric_idx.append(idx)
            else:
                str_cnts += 1
                str_idx.append(idx)
        except TypeError as e:
            print(e)

    abnormal_flag = False
    is_str = True
    candidate_values, ab_idx = None, []
    ab_threshold = 0.05
    if str_cnts == 1 or str_cnts / total_cnts <= ab_threshold:
        abnormal_flag = True
        candidate_values = column_values[numeric_idx]
        ab_idx = str_idx
        is_str = False
    elif numeric_cnts == 1 or numeric_cnts / total_cnts <= ab_threshold:
        abnormal_flag = True
        candidate_values = column_values[str_idx]
        ab_idx = numeric_idx
    return abnormal_flag, candidate_values, ab_idx, is_str
