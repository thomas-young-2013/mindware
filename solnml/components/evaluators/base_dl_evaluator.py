import os
import torch
import hashlib
import pickle as pkl

from solnml.components.utils.constants import IMG_CLS, TEXT_CLS, OBJECT_DET


def get_device(device=None):
    if device is not None:
        return device
    else:
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        # Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                torch.cuda.set_device(os.environ['CUDA_VISIBLE_DEVICES'])
            else:
                torch.cuda.set_device(0)
        return device


def get_estimator(task_type, config, max_epoch, device='cpu'):
    if task_type == IMG_CLS:
        from solnml.components.models.img_classification import _classifiers, _addons
    elif task_type == TEXT_CLS:
        from solnml.components.models.text_classification import _classifiers, _addons
    elif task_type == OBJECT_DET:
        from solnml.components.models.object_detection import _classifiers, _addons
    else:
        raise ValueError('Invalid task type %s!' % task_type)
    classifier_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    config_['random_state'] = 1
    config_['epoch_num'] = max_epoch
    config_['device'] = torch.device(device)

    new_config = dict()
    for key in config_.keys():
        if key.find(':') != -1:
            _key = key.split(':')[-1]
        else:
            _key = key
        new_config[_key] = config_[key]

    if classifier_type in _classifiers.keys():
        try:
            estimator = _classifiers[classifier_type](**new_config)
        except Exception as e:
            raise ValueError('Create estimator error: %s' % str(e))
    elif classifier_type in _addons.components.keys():
        try:
            estimator = _addons.components[classifier_type](**new_config)
        except Exception as e:
            raise ValueError('Create estimator error: %s' % str(e))
    else:
        raise ValueError('classifier type - %s is invalid!' % classifier_type)

    return classifier_type, estimator


def get_estimator_with_parameters(task_type, config, max_epoch, dataset, device='cpu', model_dir='data/dl_models/'):
    config_dict = config.get_dictionary().copy()
    _, model = get_estimator(task_type, config_dict, max_epoch, device=device)
    model_path = model_dir + TopKModelSaver.get_configuration_id(config_dict) + '.pt'
    model.set_empty_model(dataset)
    model.model.load_state_dict(torch.load(model_path))
    model.model.eval()
    return model


class TopKModelSaver(object):
    def __init__(self, k, model_dir, identifier):
        self.k = k
        self.model_dir = model_dir
        self.sorted_list_path = os.path.join(model_dir, '%s_topk_config.pkl' % identifier)

    @staticmethod
    def get_configuration_id(data_dict):
        data_list = []
        for key, value in sorted(data_dict.items(), key=lambda t: t[0]):
            if isinstance(value, float):
                value = round(value, 5)
            data_list.append('%s-%s' % (key, str(value)))
        data_id = '_'.join(data_list)
        sha = hashlib.sha1(data_id.encode('utf8'))
        return sha.hexdigest()

    @staticmethod
    def get_topk_config(config_path):
        if not os.path.exists(config_path):
            return list()
        with open(config_path, 'rb') as f:
            configs = pkl.load(f)
        return configs

    @staticmethod
    def save_topk_config(config_path, configs):
        with open(config_path, 'wb') as f:
            pkl.dump(configs, f)

    def add(self, config: dict, perf: float):
        """
            perf is larger, the better.
        :param config:
        :param perf:
        :return:
        """

        model_path_id = self.model_dir + self.get_configuration_id(config) + '.pt'
        model_path_removed = None
        save_flag, delete_flag = False, False
        sorted_list = self.get_topk_config(self.sorted_list_path)

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

        self.save_topk_config(self.sorted_list_path, sorted_list)

        return save_flag, model_path_id, delete_flag, model_path_removed
