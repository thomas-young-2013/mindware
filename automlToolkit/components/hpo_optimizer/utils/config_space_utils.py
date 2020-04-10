import random
import numpy as np
from typing import List

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    IntegerHyperparameter, FloatHyperparameter


def convert_configurations_to_array(configs: List[Configuration]) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    configuration_space = configs[0].configuration_space
    return impute_default_values(configuration_space, configs_array)


def impute_default_values(
        configuration_space: ConfigurationSpace,
        configs_array: np.ndarray
) -> np.ndarray:
    """Impute inactive hyperparameters in configuration array with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configuration_space : ConfigurationSpace

    configs_array : np.ndarray
        Array of configurations.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    for hp in configuration_space.get_hyperparameters():
        default = hp.normalized_default_value
        idx = configuration_space.get_idx_by_hyperparameter_name(hp.name)
        nonfinite_mask = ~np.isfinite(configs_array[:, idx])
        configs_array[nonfinite_mask, idx] = default

    return configs_array


# TODO: need systematical tests.
def get_hp_neighbors(hp, data_dict, num, transform=True, seed=123):
    random.seed(seed)
    if isinstance(hp, CategoricalHyperparameter):
        trans_data = hp._inverse_transform(data_dict[hp.name])
        neighbors = hp.get_neighbors(trans_data, np.random.RandomState(seed), num, transform)
        return neighbors

    lower, upper, q, name = hp.lower, hp.upper, hp.q, hp.name
    if q is None:
        if isinstance(hp, IntegerHyperparameter):
            q = 1
        if isinstance(hp, FloatHyperparameter):
            q = 1e-3

    current_value = data_dict[name]
    left_value, right_value = 0, 0
    value_list = []
    l_cnt, r_cnt = 1, 1
    l_num = num // 2
    r_num = num - l_num
    while l_cnt <= l_num and r_cnt <= r_num:
        left_value = current_value - l_cnt * q
        right_value = current_value + r_cnt * q
        if left_value < lower and abs(left_value - lower) >= q / 2:
            break
        if right_value > upper and abs(right_value - upper) >= q / 2:
            break
        value_list.append(left_value)
        value_list.append(right_value)
        l_cnt += 1
        r_cnt += 1

    num_missed = num - len(value_list)
    while l_cnt <= l_num + num_missed:
        left_value = current_value - l_cnt * q
        if left_value < lower and abs(left_value - lower) >= q / 2:
            break

        value_list.append(left_value)
        l_cnt += 1

    while r_cnt <= r_num + num_missed:
        right_value = current_value + r_cnt * q
        if right_value > upper and abs(right_value - upper) >= q / 2:
            break
        value_list.append(right_value)
        r_cnt += 1

    if not transform:
        neighbors = []
        for item in value_list:
            neighbors.append(hp._inverse_transform(item))
        return neighbors[:num]
    return value_list[:num]


def sample_hp(neighbors, seed):
    random.seed(seed)
    candidates = []
    importance_weight = reversed(list(range(1, len(neighbors) + 1)))
    for item, weight in zip(neighbors, importance_weight):
        candidates.extend([item] * weight)
    return random.choice(candidates)


# TODO: need implement it again.
def get_random_neighborhood(configuration: Configuration, num: int, seed: int) -> List[Configuration]:
    configuration_space = configuration.configuration_space
    conf_dict_data = configuration.get_dictionary()
    array_data = configuration.get_array()
    neighbor_dict = dict()
    for key, value in conf_dict_data.items():
        neighbor_dict[key] = [array_data[configuration_space._hyperparameter_idx[key]]]

    for hp in configuration.configuration_space.get_hyperparameters():
        # trans_data = hp._inverse_transform(conf_dict_data[hp.name])
        # neighbors = hp.get_neighbors(trans_data, np.random.RandomState(seed), num, False)
        # neighbor_dict[hp.name].extend(neighbors)
        if hp.name not in conf_dict_data:
            continue
        neighbors = get_hp_neighbors(hp, conf_dict_data, num, transform=False, seed=seed)
        neighbor_dict[hp.name].extend(neighbors)

    neighborhood = []
    conf_num = 0
    cnt = 0
    while conf_num < num and cnt < 5 * num:
        cnt += 1
        data = array_data.copy()
        # TODO: one exchange neighborhood
        for key in conf_dict_data.keys():
            data[configuration_space._hyperparameter_idx[key]] = random.choice(neighbor_dict[key])
            # data[configuration_space._hyperparameter_idx[key]] = sample_hp(neighbor_dict[key], seed)
        try:
            config = Configuration(configuration_space, vector=data)
            config.is_valid_configuration()
        except Exception as e:
            pass
        if config not in neighborhood:
            neighborhood.append(config)
            conf_num += 1
    assert (len(neighborhood) >= 1)
    return neighborhood


# TODO: escape the bug.
def sample_configurations(configuration_space: ConfigurationSpace, num: int) -> List[Configuration]:
    result = []
    cnt = 0
    while cnt < num:
        config = configuration_space.sample_configuration(1)
        if config not in result:
            result.append(config)
            cnt += 1
    return result


def expand_configurations(configs: List[Configuration], configuration_space: ConfigurationSpace, num: int):
    num_config = len(configs)
    num_needed = num - num_config
    config_cnt = 0
    while config_cnt < num_needed:
        config = configuration_space.sample_configuration(1)
        if config not in configs:
            configs.append(config)
            config_cnt += 1
    return configs


def get_configuration_id(data_dict):
    data_list = []
    for key, value in sorted(data_dict.items(), key=lambda t: t[0]):
        if isinstance(value, float):
            value = round(value, 5)
        data_list.append('%s-%s' % (key, str(value)))
    return '_'.join(data_list)
