from sklearn.metrics.scorer import _BaseScorer
import numpy as np
import os
import pickle as pkl

from solnml.components.utils.constants import CLS_TASKS
from solnml.components.evaluators.base_dl_evaluator import get_estimator_with_parameters
from solnml.components.ensemble.base_ensemble import BaseEnsembleModel
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

    def predict(self, data):
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

    def get_ens_model_info(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        for algo_id in self.stats["include_algorithms"]:
            model_to_eval = self.stats[algo_id]['model_to_eval']
            for idx, (node, config) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    model_path = os.path.join(self.output_dir, '%s-bagging-model%d' % (self.timestamp, model_cnt))
                    ens_config.append((algo_id, node.config, config, model_path))
                model_cnt += 1
        ens_info['ensemble_method'] = 'bagging'
        ens_info['config'] = ens_config
        return ens_info
