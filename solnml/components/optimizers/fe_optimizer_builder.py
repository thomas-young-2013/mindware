from solnml.components.optimizers.smac_optimizer import SMACOptimizer
from solnml.components.optimizers.mfse_optimizer import MfseOptimizer


def build_fe_optimizer(eval_type, evaluator, config_space,
                       per_run_time_limit=600, per_run_mem_limit=1024,
                       output_dir='./', inner_iter_num_per_iter=1, seed=1, n_jobs=1):
    if eval_type == 'partial':
        optimizer_class = MfseOptimizer
    else:
        optimizer_class = SMACOptimizer
    # return optimizer_class(task_type=task_type, input_data=input_data,
    #                        config_space=config_space,
    #                        evaluator=evaluator, model_id=model_id,
    #                        time_limit_per_trans=time_limit_per_trans,
    #                        mem_limit_per_trans=mem_limit_per_trans,
    #                        number_of_unit_resource=number_of_unit_resource,
    #                        seed=seed, n_jobs=n_jobs)

    return optimizer_class(evaluator, config_space, 'fe',
                           output_dir=output_dir,
                           per_run_time_limit=per_run_time_limit,
                           inner_iter_num_per_iter=inner_iter_num_per_iter,
                           seed=seed, n_jobs=n_jobs)
