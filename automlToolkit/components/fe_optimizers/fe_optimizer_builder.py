from automlToolkit.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer
from automlToolkit.components.fe_optimizers.multithread_evaluation_based_optimizer import \
    MultiThreadEvaluationBasedOptimizer
from automlToolkit.components.fe_optimizers.hyperband_evaluation_based_optimizer import HyperbandOptimizer


def build_fe_optimizer(eval_type, task_type, input_data, evaluator,
                       model_id: str, time_limit_per_trans: int,
                       mem_limit_per_trans: int, seed: int,
                       shared_mode: bool = False, n_jobs=4):
    if eval_type == 'partial':
        optimizer_class = HyperbandOptimizer
    elif n_jobs == 1:
        optimizer_class = EvaluationBasedOptimizer
    else:
        optimizer_class = MultiThreadEvaluationBasedOptimizer
    return optimizer_class(task_type=task_type, input_data=input_data,
                           evaluator=evaluator, model_id=model_id,
                           time_limit_per_trans=time_limit_per_trans,
                           mem_limit_per_trans=mem_limit_per_trans,
                           seed=seed, shared_mode=shared_mode, n_jobs=n_jobs)
