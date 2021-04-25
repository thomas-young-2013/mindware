import abc
import os
import time
import numpy as np
import pickle as pkl
from solnml.utils.constant import MAX_INT
from solnml.utils.logging_utils import get_logger
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.utils.topk_saver import CombinedTopKModelSaver


class BaseOptimizer(object):
    def __init__(self, evaluator: _BaseEvaluator, config_space, name, timestamp, output_dir=None, seed=None):
        self.evaluator = evaluator
        self.config_space = config_space

        assert name in ['hpo', 'fe']
        self.name = name
        self.seed = np.random.random_integers(MAX_INT) if seed is None else seed
        self.start_time = time.time()
        self.timing_list = list()
        self.incumbent = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.init_hpo_iter_num = None
        self.early_stopped_flag = False
        self.timestamp = timestamp
        self.output_dir = output_dir
        self.topk_saver = CombinedTopKModelSaver(k=50, model_dir=self.output_dir, identifier=self.timestamp)

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def iterate(self, budget=MAX_INT):
        pass

    # TODO：Refactor the other optimizers
    def update_saver(self, config_list, perf_list):
        # Check if all the configs is valid in case of storing None into the config file
        all_invalid = True

        for i, perf in enumerate(perf_list):
            if np.isfinite(perf) and perf != MAX_INT:
                all_invalid = False
                if not isinstance(config_list[i], dict):
                    config = config_list[i].get_dictionary().copy()
                else:
                    config = config_list[i].copy()
                if self.evaluator.fixed_config is not None:
                    if not isinstance(self.evaluator.fixed_config, dict):
                        fixed_config = self.evaluator.fixed_config.get_dictionary().copy()
                    else:
                        fixed_config = self.evaluator.fixed_config.copy()
                    config.update(fixed_config)
                classifier_id = config['algorithm']
                # -perf: The larger, the better.
                save_flag, model_path, delete_flag, model_path_deleted = self.topk_saver.add(config, -perf,
                                                                                             classifier_id)
                # By default, the evaluator has already stored the models.
                if save_flag:
                    pass
                else:
                    os.remove(model_path)
                    self.logger.info("Model deleted from %s" % model_path)

                try:
                    if delete_flag:
                        os.remove(model_path_deleted)
                        self.logger.info("Model deleted from %s" % model_path_deleted)
                    else:
                        pass
                except:
                    pass
            else:
                continue

        if not all_invalid:
            self.topk_saver.save_topk_config()

    def get_evaluation_stats(self):
        return

    def gc(self):
        return
