import os
import pickle as pkl
import hashlib
from abc import ABCMeta
from solnml.components.metrics.metric import get_metric
from solnml.components.utils.constants import *


def load_transformer_estimator(model_dir, config, timestamp):
    model_path = os.path.join(model_dir, '%s_%s.pkl' % (timestamp, TopKModelSaver.get_configuration_id(config)))
    with open(model_path, 'rb') as f:
        op_list, model = pkl.load(f)
    return op_list, model


def fetch_predict_estimator(task_type, config, X_train, y_train, weight_balance=0, data_balance=0, combined=False):
    # Build the ML estimator.
    from solnml.components.utils.balancing import get_weights, smote
    _fit_params = {}
    config_dict = config.get_dictionary().copy()
    if weight_balance == 1:
        _init_params, _fit_params = get_weights(
            y_train, config['estimator'], None, {}, {})
        for key, val in _init_params.items():
            config_dict[key] = val
    if data_balance == 1:
        X_train, y_train = smote(X_train, y_train)
    if task_type in CLS_TASKS:
        if combined:
            from solnml.utils.combined_evaluator import get_estimator
        else:
            from solnml.components.evaluators.cls_evaluator import get_estimator
    else:
        from solnml.components.evaluators.reg_evaluator import get_estimator
    _, estimator = get_estimator(config_dict)

    estimator.fit(X_train, y_train, **_fit_params)
    return estimator


class _BaseEvaluator(metaclass=ABCMeta):
    def __init__(self, estimator, metric, task_type,
                 evaluation_strategy, **evaluation_params):
        self.estimator = estimator
        if task_type not in TASK_TYPES:
            raise ValueError('Unsupported task type: %s' % task_type)
        self.metric = get_metric(metric)
        self.metric_name = metric
        self.evaluation_strategy = evaluation_strategy
        self.evaluation_params = evaluation_params

        if self.evaluation_strategy == 'holdout':
            if 'train_size' not in self.evaluation_params:
                self.evaluation_params['train_size']

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class TopKModelSaver(object):
    def __init__(self, k, model_dir, identifier):
        self.k = k
        self.model_dir = model_dir
        self.identifier = identifier
        self.sorted_list_path = os.path.join(model_dir, '%s_topk_config.pkl' % identifier)

    @staticmethod
    def get_configuration_id(config):
        data_dict = config.get_dictionary()
        data_list = []
        for key, value in sorted(data_dict.items(), key=lambda t: t[0]):
            if isinstance(value, float):
                value = round(value, 5)
            data_list.append('%s-%s' % (key, str(value)))
        data_id = '_'.join(data_list)
        sha = hashlib.sha1(data_id.encode('utf8'))
        return sha.hexdigest()

    def get_path_by_config(self, config, identifier):
        return os.path.join(self.model_dir, '%s_%s.pkl' % (identifier, self.get_configuration_id(config)))

    @staticmethod
    def get_topk_config(config_path):
        if not os.path.exists(config_path):
            return dict()
        with open(config_path, 'rb') as f:
            content = pkl.load(f)
        return content

    @staticmethod
    def save_topk_config(config_path, configs):
        with open(config_path, 'wb') as f:
            pkl.dump(configs, f)

    def add(self, config, perf, estimator_id):
        """
            perf: the larger, the better.
        :param estimator_id:
        :param config:
        :param perf:
        :return:
        """

        model_path_id = os.path.join(self.model_dir, '%s_%s.pkl' % (self.identifier, self.get_configuration_id(config)))
        model_path_removed = None
        save_flag, delete_flag = False, False
        sorted_dict = self.get_topk_config(self.sorted_list_path)
        sorted_list = sorted_dict.get(estimator_id, list())

        # Update existed configs
        for sorted_element in sorted_list:
            if config == sorted_element[0]:
                if perf > sorted_element[1]:
                    sorted_list.remove(sorted_element)
                    for idx, item in enumerate(sorted_list):
                        if perf > item[1]:
                            sorted_list.insert(idx, (config, perf, model_path_id))
                            break
                        if idx == len(sorted_list) - 1:
                            sorted_list.append((config, perf, model_path_id))
                            break
                    sorted_dict[estimator_id] = sorted_list
                    self.save_topk_config(self.sorted_list_path, sorted_dict)
                    return True, model_path_id, False, model_path_removed
                else:
                    return False, model_path_id, False, model_path_removed

        if len(sorted_list) == 0:
            sorted_list.append((config, perf, model_path_id))
        else:
            # Sorted list is in a descending order.
            for idx, item in enumerate(sorted_list):
                if perf > item[1]:
                    sorted_list.insert(idx, (config, perf, model_path_id))
                    break
                if idx == len(sorted_list) - 1:
                    sorted_list.append((config, perf, model_path_id))
                    break

        if len(sorted_list) > self.k:
            model_path_removed = sorted_list[-1][2]
            delete_flag = True
            sorted_list = sorted_list[:-1]
        if model_path_id in [item[2] for item in sorted_list]:
            save_flag = True

        sorted_dict[estimator_id] = sorted_list
        self.save_topk_config(self.sorted_list_path, sorted_dict)

        return save_flag, model_path_id, delete_flag, model_path_removed
