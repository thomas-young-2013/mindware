import time
from sklearn.metrics.scorer import _BaseScorer
from solnml.utils.logging_utils import get_logger


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

    def fit(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def get_ens_model_info(self):
        raise NotImplementedError

    def refit(self):
        pass
