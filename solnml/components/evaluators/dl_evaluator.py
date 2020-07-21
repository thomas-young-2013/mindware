import os
import time
import copy
import torch
import numpy as np
from multiprocessing import Lock
from math import ceil
from sklearn.metrics.scorer import accuracy_scorer

from solnml.utils.logging_utils import get_logger
from solnml.components.utils.constants import IMG_CLS
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.evaluators.base_dl_evaluator import TopKModelSaver
from solnml.components.evaluators.dl_evaluate_func import dl_holdout_validation
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_transforms
from .base_dl_evaluator import TopKModelSaver, get_estimator


class DLEvaluator(_BaseEvaluator):
    def __init__(self, clf_config, task_type, model_dir='data/dl_models/', max_epoch=150, scorer=None, dataset=None,
                 continue_training=True, device='cpu', seed=1, timestamp=None, **kwargs):
        self.hpo_config = clf_config
        self.task_type = task_type
        self.max_epoch = max_epoch
        self.scorer = scorer if scorer is not None else accuracy_scorer
        self.dataset = copy.deepcopy(dataset)
        self.continue_training = continue_training
        self.seed = seed
        self.timestamp = timestamp
        self.eval_id = 0
        self.onehot_encoder = None
        self.topk_model_saver = TopKModelSaver(k=20, model_dir=model_dir, identifier=timestamp)
        self.model_dir = model_dir
        self.device = device
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        if task_type == IMG_CLS:
            self.image_size = kwargs['image_size']

    def __call__(self, config, **kwargs):
        if self.task_type == IMG_CLS:
            data_transforms = get_transforms(config, image_size=self.image_size)
            self.dataset.load_data(data_transforms['train'], data_transforms['val'])
        else:
            self.dataset.load_data()
        start_time = time.time()

        config_dict = config.get_dictionary().copy()

        classifier_id, estimator = get_estimator(self.task_type, config_dict, self.max_epoch, device=self.device)

        epoch_ratio = kwargs.get('resource_ratio', 1.0)
        eta = kwargs.get('eta', 3)
        first_iter = kwargs.get('first_iter', False)

        config_model_path = os.path.join(self.model_dir,
                                         'tmp_' + TopKModelSaver.get_path_by_config(config, self.timestamp) + '.pt')
        if self.continue_training:
            # Continue training
            if not first_iter:
                estimator.epoch_num = ceil(estimator.epoch_num * epoch_ratio) - ceil(
                    estimator.epoch_num * epoch_ratio / eta)
                estimator.load_path = config_model_path
            else:
                estimator.epoch_num = ceil(estimator.epoch_num * epoch_ratio)
        else:
            estimator.epoch_num = ceil(estimator.epoch_num * epoch_ratio)

        if 'profile_epoch' in kwargs or 'profile_iter' in kwargs:  # Profile mode
            try:
                time_cost = dl_holdout_validation(estimator, self.scorer, self.dataset, random_state=self.seed,
                                                  **kwargs)
            except Exception as e:
                self.logger.error(e)
                time_cost = np.inf
            self.logger.info('%d-Evaluation<%s> | Profile time cost: %.2f seconds' %
                             (self.eval_id, classifier_id, time_cost))
            return time_cost

        try:
            score = dl_holdout_validation(estimator, self.scorer, self.dataset, random_state=self.seed, **kwargs)
        except Exception as e:
            self.logger.error(e)
            score = -np.inf
        self.logger.info('%d-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds' %
                         (self.eval_id, classifier_id,
                          self.scorer._sign * score,
                          time.time() - start_time))
        self.eval_id += 1

        # Save low-resource models
        if self.continue_training and np.isfinite(score) and epoch_ratio != 1.0:
            state = {'model': estimator.model.state_dict(),
                     'optimizer': estimator.optimizer_.state_dict(),
                     'scheduler': estimator.scheduler.state_dict(),
                     'epoch_num': estimator.epoch_num}
            torch.save(state, config_model_path)

        # Save top K models with the largest validation scores.
        if 'rw_lock' not in kwargs and kwargs['rw_lock'] is None:
            self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
        lock = kwargs.get('rw_lock', Lock())
        lock.acquire()
        if np.isfinite(score):
            save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(config, score)
            if save_flag is True:
                state = {'model': estimator.model.state_dict(),
                         'optimizer': estimator.optimizer_.state_dict(),
                         'scheduler': estimator.scheduler.state_dict(),
                         'epoch_num': estimator.epoch_num}
                torch.save(state, model_path)
                self.logger.info("Model saved to %s" % model_path)

            # In case of double-deletion
            try:
                if delete_flag and os.path.exists(model_path_deleted):
                    os.remove(model_path_deleted)
                    self.logger.info("Model deleted from %s" % model_path)
            except:
                pass
        lock.release()

        # Turn it into a minimization problem.
        return -score
