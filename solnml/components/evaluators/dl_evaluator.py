import os
import time
import torch
import numpy as np
from sklearn.metrics.scorer import accuracy_scorer

from solnml.utils.logging_utils import get_logger
from solnml.components.utils.constants import IMG_CLS
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.evaluators.dl_evaluate_func import dl_holdout_validation
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_transforms
from .base_dl_evaluator import TopKModelSaver, get_estimator


class DLEvaluator(_BaseEvaluator):
    def __init__(self, clf_config, task_type, model_dir='data/dl_models/', scorer=None, dataset=None, device='cpu',
                 seed=1):
        self.hpo_config = clf_config
        self.task_type = task_type
        self.scorer = scorer if scorer is not None else accuracy_scorer
        self.dataset = dataset
        self.seed = seed
        self.eval_id = 0
        self.onehot_encoder = None
        self.topk_model_saver = TopKModelSaver(k=20, model_dir=model_dir)
        self.model_dir = model_dir
        self.device = device
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

    def __call__(self, config, **kwargs):
        if self.task_type == IMG_CLS:
            data_transforms = get_transforms(config)
            self.dataset.set_udf_transform(data_transforms)

        self.dataset.load_data()
        start_time = time.time()

        config_dict = config.get_dictionary().copy()
        classifier_id, clf = get_estimator(self.task_type, config_dict, device=self.device)
        try:
            score = dl_holdout_validation(clf, self.scorer, self.dataset, random_state=self.seed)
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
