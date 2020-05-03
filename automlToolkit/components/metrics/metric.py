from sklearn.metrics.scorer import make_scorer, _BaseScorer
from functools import partial


def get_metric(metric):
    # Metrics for classification
    if metric in ["accuracy", "acc"]:
        from sklearn.metrics import accuracy_score
        return make_scorer(accuracy_score)
    elif metric in ["balanced_accuracy", "bal_acc"]:
        from sklearn.metrics.scorer import balanced_accuracy_scorer
        return balanced_accuracy_scorer
    elif metric == 'f1':
        from sklearn.metrics import f1_score
        return make_scorer(partial(f1_score, average='macro'))
    elif metric == 'precision':
        from sklearn.metrics import precision_score
        return make_scorer(precision_score)
    elif metric == 'recall':
        from sklearn.metrics import recall_score
        return make_scorer(recall_score)
    elif metric == "auc":
        from sklearn.metrics import roc_auc_score
        return make_scorer(roc_auc_score, needs_threshold=True)
    elif metric in ['log_loss', 'cross_entropy']:
        from sklearn.metrics import log_loss
        return make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    # Metrics for regression
    elif metric in ["mean_squared_error", "mse"]:
        from sklearn.metrics import mean_squared_error
        return make_scorer(mean_squared_error, greater_is_better=False)
    elif metric == "rmse":
        from .rgs_metrics import rmse
        return make_scorer(rmse, greater_is_better=False)
    elif metric in ['mean_squared_log_error', "msle"]:
        from sklearn.metrics import mean_squared_log_error
        return make_scorer(mean_squared_log_error, greater_is_better=False)
    elif metric == "evs":
        from sklearn.metrics import explained_variance_score
        return make_scorer(explained_variance_score)
    elif metric == "r2":
        from sklearn.metrics import r2_score
        return make_scorer(r2_score)
    elif metric == "max_error":
        from sklearn.metrics import max_error
        return make_scorer(max_error, greater_is_better=False)
    elif metric in ["mean_absolute_error", "mae"]:
        from sklearn.metrics import mean_absolute_error
        return make_scorer(mean_absolute_error, greater_is_better=False)
    elif metric == "median_absolute_error":
        from sklearn.metrics import median_absolute_error
        return make_scorer(median_absolute_error, greater_is_better=False)
    elif isinstance(metric, _BaseScorer):
        return metric
    elif isinstance(metric, callable):
        import warnings
        warnings.warn("metric receives a callable and we consider to maximize it!")
        return make_scorer(metric)
    else:
        raise ValueError("Given", str(metric), ". Expect a str or a sklearn.Scorer or a callable")
