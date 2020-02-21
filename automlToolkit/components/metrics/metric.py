from typing import Callable
from sklearn.metrics.scorer import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics.scorer import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Standard regression scores
reg_metric_names = ['r2_score', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error']
reg_metrics = [r2_score, mean_squared_error, mean_absolute_error, median_absolute_error]
REGRESSION_METRICS = dict(zip(reg_metric_names, reg_metrics))

# Standard Classification Scores
cls_metric_names = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
cls_metrics = [accuracy_score, f1_score, roc_auc_score, precision_score, recall_score]
CLASSIFICATION_METRICS = dict(zip(cls_metric_names, cls_metrics))


def fetch_scorer(_scorer: [str, Callable], task_type: str):
    if isinstance(_scorer, str):
        if task_type == 'classification' and _scorer in CLASSIFICATION_METRICS.keys():
            return CLASSIFICATION_METRICS[_scorer]
        elif task_type == 'regression' and _scorer in REGRESSION_METRICS.keys():
            return REGRESSION_METRICS[_scorer]
    elif isinstance(_scorer, Callable):
        return _scorer
    else:
        raise ValueError('Unsupported metric.')
