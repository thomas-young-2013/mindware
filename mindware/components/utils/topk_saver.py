import os
import pickle as pkl
import hashlib


def load_combined_transformer_estimator(model_dir, config, timestamp):
    model_path = os.path.join(model_dir, '%s_%s.pkl' % (timestamp, CombinedTopKModelSaver.get_configuration_id(config)))
    with open(model_path, 'rb') as f:
        op_list, model, _ = pkl.load(f)
    return op_list, model


class BaseTopKModelSaver(object):
    def __init__(self, k, model_dir, identifier):
        self.k = k
        self.model_dir = model_dir
        self.identifier = identifier
        self.sorted_list_path = os.path.join(model_dir, '%s_topk_config.pkl' % identifier)
        self.sorted_dict = None

    @staticmethod
    def get_topk_config(config_path):
        if not os.path.exists(config_path):
            return dict()
        with open(config_path, 'rb') as f:
            content = pkl.load(f)
        return content

    def save_topk_config(self):
        with open(self.sorted_list_path, 'wb') as f:
            pkl.dump(self.sorted_dict, f)


class CombinedTopKModelSaver(BaseTopKModelSaver):
    @staticmethod
    def get_configuration_id(config: dict):
        _config = sorted(config.items(), key=lambda x: x[0])
        data_dict = dict(_config)
        data_list = []
        for key, value in sorted(data_dict.items(), key=lambda t: t[0]):
            if isinstance(value, float):
                value = round(value, 5)
            data_list.append('%s-%s' % (key, str(value)))
        data_id = '_'.join(data_list)
        sha = hashlib.sha1(data_id.encode('utf8'))
        return sha.hexdigest()

    @staticmethod
    def get_path_by_config(output_dir, config, identifier):
        return os.path.join(output_dir, '%s_%s.pkl' % (identifier, CombinedTopKModelSaver.get_configuration_id(config)))

    def add(self, config, perf, estimator_id):
        """
            perf: the larger, the better.
        :param estimator_id:
        :param config:
        :param perf:
        :return:
        """
        _config = config.copy()
        config = dict(sorted(_config.items(), key=lambda x: x[0]))
        model_path_id = self.get_path_by_config(self.model_dir, config, self.identifier)
        model_path_removed = None
        save_flag, delete_flag = False, False
        self.sorted_dict = self.get_topk_config(self.sorted_list_path)
        sorted_list = self.sorted_dict.get(estimator_id, list())

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
                    self.sorted_dict[estimator_id] = sorted_list
                return True, model_path_id, False, model_path_removed

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

        self.sorted_dict[estimator_id] = sorted_list

        return save_flag, model_path_id, delete_flag, model_path_removed
