import os
import torch
import hashlib
import pickle as pkl

from mindware.components.utils.constants import IMG_CLS, TEXT_CLS, OBJECT_DET
from mindware.components.utils.topk_saver import CombinedTopKModelSaver


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
        from mindware.components.models.img_classification import _classifiers, _addons
    elif task_type == TEXT_CLS:
        from mindware.components.models.text_classification import _classifiers, _addons
    elif task_type == OBJECT_DET:
        from mindware.components.models.object_detection import _classifiers, _addons
    else:
        raise ValueError('Invalid task type %s!' % task_type)
    classifier_type = config['algorithm']
    config_ = config.copy()
    config_.pop('algorithm', None)
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


def get_estimator_with_parameters(task_type, config, max_epoch, dataset, timestamp, device='cpu',
                                  model_dir='data/dl_models/'):
    if not isinstance(config, dict):
        config_dict = config.get_dictionary().copy()
    else:
        config_dict = config.copy()
    _, model = get_estimator(task_type, config_dict, max_epoch, device=device)
    model_path = CombinedTopKModelSaver.get_path_by_config(model_dir, config, timestamp)
    model.set_empty_model(config=config, dataset=dataset)
    model.model.load_state_dict(torch.load(model_path)['model'])
    model.model.eval()
    return model
