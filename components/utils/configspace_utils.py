import random
from typing import List
from ConfigSpace import Configuration, ConfigurationSpace


def sample_configurations(configuration_space: ConfigurationSpace,
                          sample_size: int, historical_configs: List[Configuration], seed=1):
    configuration_space.seed(seed)
    result = []
    sample_cnt = 0
    if len(historical_configs) == 0:
        result.append(configuration_space.get_default_configuration())

    while len(result) < sample_size:
        config = configuration_space.sample_configuration(1)
        if config not in result and config not in historical_configs:
            result.append(config)
        sample_cnt += 1
        if sample_cnt > 50 * sample_size:
            break

    # if len(result) == 0:
    #     hist_num = len(historical_configs)
    #     if hist_num > sample_size:
    #         idxs = random.sample(range(len(historical_configs)), sample_size)
    #         result = [historical_configs[idx] for idx in idxs]
    #     else:
    #         result = historical_configs.copy()
    return result


def check_true(p):
    if p in ("True", "true", 1, True):
        return True
    return False


def check_false(p):
    if p in ("False", "false", 0, False):
        return True
    return False


def check_none(p):
    if p in ("None", "none", None):
        return True
    return False


def check_for_bool(p):
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
        raise ValueError("%s is not a bool" % str(p))
