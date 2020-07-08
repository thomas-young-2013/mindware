from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, Constant


def parse_set(string, hp_type):
    choices = list(map(lambda x: hp_type(x), string[1:-1].split(',')))
    return choices


def parse_list(string, hp_type):
    bounds = string[1:-1].split(',')
    if len(bounds) != 2:
        raise ValueError("Invalid bound for uniform hyperparameter")
    bounds = list(map(lambda x: hp_type(x), bounds))
    return bounds


class ConfigParser:
    def __init__(self, logger=None):
        self.logger = logger

    def read_key_values_from_file(self, filename, delimiter=' '):
        key_values = dict()

        if filename is None:
            self.logger.info("Config file %s not found!" % filename)
            return key_values

        # open the config file
        with open(filename, "r") as configfile:
            for line in configfile:
                elements = list(map(lambda x: x.strip(), line.split(delimiter)))
                key_values[(elements[0], elements[1])] = elements[2:]
        return key_values

    def read(self, filename, key_values_dict=None):
        key_values_dict = key_values_dict if key_values_dict else self.read_key_values_from_file(filename)

        if key_values_dict is None:
            self.logger.info('Nothing to parse for ConfigParser!')

        hp_space_dict = dict()
        for (estimator_id, hp_name), values in key_values_dict.items():
            if estimator_id not in hp_space_dict:
                hp_space_dict[estimator_id] = ConfigurationSpace()
            range = values[0]

            # Categorical
            if range[0] == '{':
                hp_type = values[1]
                if hp_type == 'int':
                    hp_type = int
                elif hp_type == 'float':
                    hp_type = float
                else:
                    hp_type = str
                choices = parse_set(range, hp_type)
                default_value = hp_type(values[2])
                hp = CategoricalHyperparameter(hp_name, choices, default_value=default_value)
            # Uniform
            elif range[0] == '[':
                hp_type = values[1]
                if hp_type == 'int':
                    hp_type = int
                elif hp_type == 'float':
                    hp_type = float
                else:
                    raise ValueError("Invalid type %s for uniform hyperparameter!" % hp_type)
                bounds = parse_list(range, hp_type)
                default_value = values[2]
                if len(values) > 3:
                    if_log = values[3]
                else:
                    if_log = False
                if hp_type == 'int':
                    hp = UniformIntegerHyperparameter(hp_name, lower=bounds[0], upper=bounds[1],
                                                      default_value=default_value, log=if_log)
                else:
                    hp = UniformFloatHyperparameter(hp_name, lower=bounds[0], upper=bounds[1],
                                                    default_value=default_value, log=if_log)
            # Constant
            else:
                hp_type = values[1]
                if hp_type == 'int':
                    hp = Constant(hp_name, int(range))
                elif hp_type == 'float':
                    hp = Constant(hp_name, float(range))
                else:
                    hp = Constant(hp_name, range)
            hp_space_dict[estimator_id].add_hyperparameter(hp)

        return hp_space_dict
