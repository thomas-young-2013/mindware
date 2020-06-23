import os
import time
import torch
from sklearn.metrics.scorer import _BaseScorer
from torch.utils.data import Dataset
from solnml.utils.logging_utils import get_logger
from solnml.datasets.base_dataset import BaseDataset
from solnml.components.evaluators.base_dl_evaluator import TopKModelSaver, get_estimator


class BaseEnsembleModel(object):
    """Base class for model ensemble"""

    def __init__(self, stats, ensemble_method: str,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None):
        self.stats = stats
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.output_dir = output_dir

        self.train_predictions = list()
        self.train_labels = None
        self.seed = self.stats['split_seed']
        self.timestamp = str(time.time())
        logger_name = 'EnsembleBuilder'
        self.logger = get_logger(logger_name)

    def fit(self, data: BaseDataset):
        raise NotImplementedError

    def predict(self, dataset: Dataset, sampler=None):
        raise NotImplementedError

    def refit(self, dataset: BaseDataset):
        for algo_id in self.stats['include_algorithms']:
            for config in self.stats[algo_id]:
                config_dict = config.get_dictionary().copy()
                model_path = self.output_dir + TopKModelSaver.get_configuration_id(config_dict) + '.pt'
                # Remove the old models.
                if os.path.exists(model_path):
                    os.remove(model_path)

                # Refit the models.
                _, clf = get_estimator(config_dict)
                # TODO: if train ans val are two parts, we need to merge it into one dataset.
                clf.fit(dataset.train_dataset)
                # Save to the disk.
                torch.save(clf.model.state_dict(), model_path)
