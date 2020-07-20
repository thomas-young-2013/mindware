import numpy as np
from sklearn.metrics.scorer import _BaseScorer

from solnml.components.utils.constants import CLS_TASKS, IMG_CLS
from solnml.datasets.base_dl_dataset import DLDataset
from solnml.components.evaluators.base_dl_evaluator import get_estimator_with_parameters
from solnml.components.ensemble.dl_ensemble.base_ensemble import BaseEnsembleModel
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_test_transforms

from functools import reduce


class Bagging(BaseEnsembleModel):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int,
                 max_epoch: int,
                 metric: _BaseScorer,
                 timestamp: float,
                 output_dir=None,
                 device='cpu', **kwargs):
        super().__init__(stats=stats,
                         ensemble_method='bagging',
                         ensemble_size=ensemble_size,
                         task_type=task_type,
                         max_epoch=max_epoch,
                         metric=metric,
                         timestamp=timestamp,
                         output_dir=output_dir,
                         device=device)

        if self.task_type == IMG_CLS:
            self.image_size = kwargs['image_size']

    def fit(self, train_data):
        # Do nothing, models has been trained and saved.
        return self

    def predict(self, test_data: DLDataset, mode='test'):
        model_pred_list = list()
        final_pred = list()

        model_cnt = 0
        for algo_id in self.stats["include_algorithms"]:
            model_configs = self.stats[algo_id]['model_configs']
            for idx, config in enumerate(model_configs):
                if self.task_type == IMG_CLS:
                    test_transforms = get_test_transforms(config, image_size=self.image_size)
                    test_data.load_test_data(test_transforms)
                    test_data.load_data(test_transforms, test_transforms)
                else:
                    test_data.load_test_data()
                    test_data.load_data()

                if mode == 'test':
                    dataset = test_data.test_dataset
                else:
                    if test_data.subset_sampler_used:
                        dataset = test_data.train_dataset
                    else:
                        dataset = test_data.val_dataset
                estimator = get_estimator_with_parameters(self.task_type, config, self.max_epoch,
                                                          dataset, self.timestamp, device=self.device)
                if self.task_type in CLS_TASKS:
                    if mode == 'test':
                        model_pred_list.append(estimator.predict_proba(test_data.test_dataset))
                    else:
                        if test_data.subset_sampler_used:
                            model_pred_list.append(
                                estimator.predict_proba(test_data.train_dataset, sampler=test_data.val_sampler))
                        else:
                            model_pred_list.append(estimator.predict_proba(test_data.val_dataset))
                else:
                    if mode == 'test':
                        model_pred_list.append(estimator.predict(test_data.test_dataset))
                    else:
                        if test_data.subset_sampler_used:
                            model_pred_list.append(
                                estimator.predict(test_data.train_dataset, sampler=test_data.val_sampler))
                        else:
                            model_pred_list.append(estimator.predict(test_data.val_dataset))
                model_cnt += 1

        # Calculate the average of predictions
        for i in range(len(model_pred_list[0])):
            sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
            pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
            final_pred.append(pred_average)

        return np.array(final_pred)

    def get_ens_model_info(self):
        raise NotImplementedError
