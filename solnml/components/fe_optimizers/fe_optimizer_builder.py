from solnml.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer
from solnml.components.fe_optimizers.multithread_evaluation_based_optimizer import \
    MultiThreadEvaluationBasedOptimizer
from solnml.components.fe_optimizers.hyperband_evaluation_based_optimizer import HyperbandOptimizer
from solnml.components.fe_optimizers.bo_optimizer import BayesianOptimizationOptimizer
from solnml.components.fe_optimizers.ano_bo_optimizer import AnotherBayesianOptimizationOptimizer
from solnml.components.fe_optimizers.mfse_optimizer import MfseOptimizer


def build_fe_optimizer(algo, eval_type, task_type, config_space, input_data, evaluator,
                       model_id: str, time_limit_per_trans: int,
                       mem_limit_per_trans: int, seed: int,
                       number_of_unit_resource: int = 1,
                       shared_mode: bool = False, n_jobs=4):
    if eval_type == 'partial':
        optimizer_class = MfseOptimizer
    else:
        optimizer_class = AnotherBayesianOptimizationOptimizer
        # optimizer_class = BayesianOptimizationOptimizer
    return optimizer_class(task_type=task_type, input_data=input_data,
                           config_space=config_space,
                           evaluator=evaluator, model_id=model_id,
                           time_limit_per_trans=time_limit_per_trans,
                           mem_limit_per_trans=mem_limit_per_trans,
                           number_of_unit_resource=number_of_unit_resource,
                           seed=seed, n_jobs=n_jobs)
