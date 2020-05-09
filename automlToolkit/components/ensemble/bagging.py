from sklearn.metrics.scorer import _BaseScorer
import numpy as np
import os
import pickle as pkl

from automlToolkit.components.utils.constants import CLS_TASKS
from automlToolkit.components.evaluators.base_evaluator import fetch_predict_estimator
from automlToolkit.components.ensemble.base_ensemble import BaseEnsembleModel
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

    def fit(self, datanode):
        model_cnt = 0
        for algo_id in self.stats["include_algorithms"]:
            train_list = self.stats[algo_id]['train_data_list']
            configs = self.stats[algo_id]['configurations']
            for idx in range(len(train_list)):
                X, y = train_list[idx].data
                for _config in configs:
                    if self.base_model_mask[model_cnt] == 1:
                        estimator = fetch_predict_estimator(self.task_type, _config, X, y,
                                                            weight_balance=train_list[idx].enable_balance,
                                                            data_balance=train_list[idx].data_balance)
                        with open(os.path.join(self.output_dir, '%s-bagging-model%d' % (self.timestamp, model_cnt)), 'wb') as f:
                            pkl.dump(estimator, f)
                    model_cnt += 1
        return self

    def predict(self, data, solvers):
        model_pred_list = []
        final_pred = []
        # Get predictions from each model
        model_cnt = 0
        for algo_id in self.stats["include_algorithms"]:
            train_list = self.stats[algo_id]['train_data_list']
            configs = self.stats[algo_id]['configurations']
            for train_node in train_list:
                test_node = solvers[algo_id].optimizer['fe'].apply(data, train_node)
                for _ in configs:
                    if self.base_model_mask[model_cnt] == 1:
                        with open(os.path.join(self.output_dir, '%s-bagging-model%d' % (self.timestamp, model_cnt)), 'rb') as f:
                            estimator = pkl.load(f)
                            if self.task_type in CLS_TASKS:
                                model_pred_list.append(estimator.predict_proba(test_node.data[0]))
                            else:
                                model_pred_list.append(estimator.predict(test_node.data[0]))
                    model_cnt += 1

        # Calculate the average of predictions
        for i in range(len(data.data[0])):
            sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
            pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
            final_pred.append(pred_average)

        return np.array(final_pred)
