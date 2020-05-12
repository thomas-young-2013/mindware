from solnml.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from solnml.components.hpo_optimizer.mfse_optimizer import MfseOptimizer


def build_hpo_optimizer(eval_type, evaluator, config_space,
                        per_run_time_limit=600, per_run_mem_limit=1024,
                        output_dir='./', trials_per_iter=1, seed=1, n_jobs=1):
    if eval_type == 'partial':
        optimizer_class = MfseOptimizer
    else:
        # TODO: Support asynchronous BO
        optimizer_class = SMACOptimizer
    return optimizer_class(evaluator, config_space,
                           output_dir=output_dir,
                           per_run_time_limit=per_run_time_limit,
                           trials_per_iter=trials_per_iter,
                           seed=seed, n_jobs=n_jobs)
