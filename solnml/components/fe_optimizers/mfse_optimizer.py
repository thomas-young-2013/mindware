import time
import os
import numpy as np

from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.hpo_optimizer.base.mfsebase import MfseBase
from solnml.components.fe_optimizers.ano_bo_optimizer import AnotherBayesianOptimizationOptimizer
from solnml.components.hpo_optimizer.base_optimizer import MAX_INT


class MfseOptimizer(AnotherBayesianOptimizationOptimizer, MfseBase):
    def __init__(self, task_type, input_data: DataNode,
                 config_space, evaluator: _BaseEvaluator,
                 model_id: str, time_limit_per_trans: int,
                 mem_limit_per_trans: int,
                 seed: int, n_jobs=1,
                 number_of_unit_resource=1,
                 time_budget=600, inner_iter_num_per_iter=1,
                 R=27, eta=3):
        AnotherBayesianOptimizationOptimizer.__init__(self, task_type=task_type, input_data=input_data,
                                                      config_space=config_space,
                                                      evaluator=evaluator, model_id=model_id,
                                                      time_limit_per_trans=time_limit_per_trans,
                                                      mem_limit_per_trans=mem_limit_per_trans,
                                                      seed=seed, n_jobs=n_jobs,
                                                      number_of_unit_resource=number_of_unit_resource,
                                                      time_budget=time_budget)
        MfseBase.__init__(self, eval_func=self.evaluator, config_space=self.hyperparameter_space,
                          seed=seed, R=R, eta=eta, n_jobs=n_jobs)

        self.inner_iter_num_per_iter = inner_iter_num_per_iter

    def iterate(self, budget=MAX_INT):
        '''
            Iterate a SH procedure (inner loop) in Hyperband.
        :return:
        '''
        _start_time = time.time()
        for _ in range(self.inner_iter_num_per_iter):
            _time_elapsed = time.time() - _start_time
            if _time_elapsed >= budget:
                break
            budget_left = budget - _time_elapsed
            self._iterate(self.s_values[self.inner_iter_id], budget=budget_left)
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.s_max + 1)

            # Remove tmp model
            if self.evaluator.continue_training:
                for filename in os.listdir(self.evaluator.model_dir):
                    # Temporary model
                    if 'tmp_%s' % self.evaluator.timestamp in filename:
                        try:
                            filepath = os.path.join(self.evaluator.model_dir, filename)
                            os.remove(filepath)
                        except Exception:
                            pass

        if len(self.incumbent_perfs) > 0:
            inc_idx = np.argmin(np.array(self.incumbent_perfs))

            for idx in range(len(self.incumbent_perfs)):
                if hasattr(self.evaluator, 'fe_config'):
                    fe_config = self.evaluator.fe_config
                else:
                    fe_config = None
                self.eval_dict[(fe_config, self.incumbent_configs[idx])] = [-self.incumbent_perfs[idx], time.time()]

            self.incumbent_perf = -self.incumbent_perfs[inc_idx]
            self.incumbent_config = self.incumbent_configs[inc_idx]

        self.perfs = self.incumbent_perfs
        self.configs = self.incumbent_configs

        # Incumbent performance: the large, the better.
        iteration_cost = time.time() - _start_time
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def get_evaluation_stats(self):
        return self.evaluation_stats
