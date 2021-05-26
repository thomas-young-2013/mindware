from solnml.components.optimizers.smac_optimizer import SMACOptimizer
from solnml.components.optimizers.random_search_optimizer import RandomSearchOptimizer
from solnml.components.optimizers.mfse_optimizer import MfseOptimizer
from solnml.components.optimizers.bohb_optimizer import BohbOptimizer
from solnml.components.optimizers.tpe_optimizer import TPEOptimizer


def build_hpo_optimizer(eval_type, evaluator, config_space, optimizer='smac',
                        per_run_time_limit=600, per_run_mem_limit=1024,
                        output_dir='./', inner_iter_num_per_iter=1,
                        timestamp=None, seed=1, n_jobs=1):
    if eval_type == 'partial':
        optimizer_class = MfseOptimizer
    elif eval_type == 'partial_bohb':
        optimizer_class = BohbOptimizer
    else:
        # TODO: Support asynchronous BO
        if optimizer == 'random_search':
            optimizer_class = RandomSearchOptimizer
        elif optimizer == 'tpe':
            optimizer_class = TPEOptimizer
        elif optimizer == 'smac':
            optimizer_class = SMACOptimizer
        else:
            raise ValueError("Invalid optimizer %s" % optimizer)

    return optimizer_class(evaluator, config_space, 'hpo',
                           eval_type=eval_type, output_dir=output_dir,
                           per_run_time_limit=per_run_time_limit,
                           inner_iter_num_per_iter=inner_iter_num_per_iter,
                           timestamp=timestamp, seed=seed, n_jobs=n_jobs)
