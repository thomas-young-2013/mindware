import os
import torch
import hashlib

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


def get_estimator(task_type, config, device='cpu'):
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
    config_['device'] = torch.device(device)
    try:
        estimator = _classifiers[classifier_type](**config_)
    except:
        estimator = _addons.components[classifier_type](**config_)
    return classifier_type, estimator


def get_estimator_with_parameters(task_type, config, dataset, device='cpu', model_dir='data/dl_models/'):
    config_dict = config.get_dictionary().copy()
    _, model = get_estimator(task_type, config_dict, device=device)
    model_path = model_dir + TopKModelSaver.get_configuration_id(config_dict) + '.pt'
    model.set_empty_model(dataset)
    model.model.load_state_dict(torch.load(model_path))
    model.model.eval()
    return model


class TopKModelSaver(object):
    def __init__(self, k, model_dir):
        self.k = k
        self.sorted_list = list()
        self.model_dir = model_dir

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

        if len(self.sorted_list) == 0:
            self.sorted_list.append((config, perf, model_path_id))
        else:
            # Sorted list is in a descending order.
            for idx, item in enumerate(self.sorted_list):
                if perf > item[1]:
                    self.sorted_list.insert(idx, (config, perf, model_path_id))
                    break

        if len(self.sorted_list) > self.k:
            model_path_removed = self.sorted_list[-1][2]
            delete_flag = True
            self.sorted_list = self.sorted_list[:-1]
        if model_path_id in [item[2] for item in self.sorted_list]:
            save_flag = True
        return save_flag, model_path_id, delete_flag, model_path_removed
