import os
import numpy as np
from sklearn.metrics.scorer import _BaseScorer

from solnml.datasets.base_dataset import BaseDataset
from solnml.components.utils.constants import CLS_TASKS
from solnml.components.evaluators.base_dl_evaluator import get_estimator_with_parameters
from solnml.components.ensemble.dl_ensemble.base_ensemble import BaseEnsembleModel
from functools import reduce


class Bagging(BaseEnsembleModel):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None):
        super().__init__(stats=stats,
                         ensemble_method='bagging',
                         ensemble_size=ensemble_size,
                         task_type=task_type,
                         metric=metric,
                         output_dir=output_dir)

    def fit(self, train_data):
        # Do nothing, models has been trained and saved.
        return self

    def predict(self, data: BaseDataset):
        model_pred_list = list()
        final_pred = list()

        model_cnt = 0
        for algo_id in self.stats["include_algorithms"]:
            model_configs = self.stats[algo_id]['model_configs']
            for idx, config in enumerate(model_configs):
                estimator = get_estimator_with_parameters(config, self.output_dir)
                if self.task_type in CLS_TASKS:
                    model_pred_list.append(estimator.predict_proba(data))
                else:
                    model_pred_list.append(estimator.predict(data))
                model_cnt += 1

        # Calculate the average of predictions
        for i in range(len(data.data[0])):
            sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
            pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
            final_pred.append(pred_average)

        return np.array(final_pred)

