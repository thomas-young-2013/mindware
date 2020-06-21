import os
import time
import torch
import hashlib
import numpy as np
from sklearn.metrics.scorer import accuracy_scorer

from solnml.utils.logging_utils import get_logger
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.evaluators.img_evaluate_func import img_holdout_validation
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_transforms


def get_estimator(config):
    from solnml.components.models.img_classification import _classifiers, _addons
    classifier_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    config_['random_state'] = 1
    try:
        estimator = _classifiers[classifier_type](**config_)
    except:
        estimator = _addons.components[classifier_type](**config_)
    return classifier_type, estimator


def get_estimator_with_parameters(config, model_dir='data/dl_models/'):
    config_dict = config.get_dictionary().copy()
    _, model = get_estimator(config_dict)
    model_path = model_dir + TopKModelSaver.get_configuration_id(config_dict) + '.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
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


class ImgClassificationEvaluator(_BaseEvaluator):
    def __init__(self, clf_config, model_dir='data/dl_models/', scorer=None, dataset=None, seed=1):
        self.hpo_config = clf_config
        self.scorer = scorer if scorer is not None else accuracy_scorer
        self.dataset = dataset
        self.seed = seed
        self.eval_id = 0
        self.onehot_encoder = None
        self.topk_model_saver = TopKModelSaver(k=20, model_dir=model_dir)
        self.model_dir = model_dir
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

    def __call__(self, config, **kwargs):
        data_transforms = get_transforms(config)

        self.dataset.set_udf_transform(data_transforms)
        self.dataset.load_data()
        start_time = time.time()


        config_dict = config.get_dictionary().copy()
        classifier_id, clf = get_estimator(config_dict)

        try:
            score = img_holdout_validation(clf, self.scorer, self.dataset, random_state=self.seed)
        except Exception as e:
            self.logger.error(e)
            score = -np.inf

        self.logger.debug('%d-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds' %
                          (self.eval_id, classifier_id,
                           self.scorer._sign * score,
                           time.time() - start_time))
        self.eval_id += 1

        # Save top K models with the largest validation scores.
        if np.isfinite(score):
            save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(config_dict, score)
            if save_flag is True:
                torch.save(clf.model.state_dict(), model_path)

            if delete_flag is True:
                os.remove(model_path_deleted)

        # Turn it into a minimization problem.
        return -score
