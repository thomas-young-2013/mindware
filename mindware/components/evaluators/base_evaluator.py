from abc import ABCMeta
from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import *


def fetch_predict_estimator(task_type, estimator_id, config, X_train, y_train, weight_balance=0, data_balance=0):
    # Build the ML estimator.
    from mindware.components.utils.balancing import get_weights, smote
    _fit_params = {}
    config_dict = config.copy()
    if weight_balance == 1:
        _init_params, _fit_params = get_weights(
            y_train, estimator_id, None, {}, {})
        for key, val in _init_params.items():
            config_dict[key] = val
    if data_balance == 1:
        X_train, y_train = smote(X_train, y_train)
    if task_type in CLS_TASKS:
        from mindware.components.evaluators.cls_evaluator import get_estimator
    elif task_type in RGS_TASKS:
        from mindware.components.evaluators.rgs_evaluator import get_estimator
    _, estimator = get_estimator(config_dict, estimator_id)

    estimator.fit(X_train, y_train, **_fit_params)
    return estimator


class _BaseEvaluator(metaclass=ABCMeta):
    def __init__(self, estimator, metric, task_type,
                 evaluation_strategy, **evaluation_params):
        self.estimator = estimator
        if task_type not in TASK_TYPES:
            raise ValueError('Unsupported task type: %s' % task_type)
        self.metric = get_metric(metric)
        self.metric_name = metric
        self.evaluation_strategy = evaluation_strategy
        self.evaluation_params = evaluation_params

        if self.evaluation_strategy == 'holdout':
            if 'train_size' not in self.evaluation_params:
                self.evaluation_params['train_size']

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
