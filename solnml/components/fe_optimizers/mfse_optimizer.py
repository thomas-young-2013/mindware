import time
import numpy as np

from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.hpo_optimizer.base.mfsebase import MfseBase
from solnml.components.fe_optimizers.bo_optimizer import BayesianOptimizationOptimizer


class MfseOptimizer(BayesianOptimizationOptimizer, MfseBase):
    def __init__(self, task_type, input_data: DataNode, evaluator: _BaseEvaluator,
                 model_id: str, time_limit_per_trans: int,
                 mem_limit_per_trans: int,
                 seed: int, n_jobs=1,
                 number_of_unit_resource=1,
                 time_budget=600,
                 R=81, eta=3):
        BayesianOptimizationOptimizer.__init__(self, task_type=task_type, input_data=input_data,
                                               evaluator=evaluator, model_id=model_id,
                                               time_limit_per_trans=time_limit_per_trans,
                                               mem_limit_per_trans=mem_limit_per_trans,
                                               seed=seed, n_jobs=n_jobs,
                                               number_of_unit_resource=number_of_unit_resource,
                                               time_budget=time_budget)
        MfseBase.__init__(self, eval_func=self.evaluate_function, config_space=self.hyperparameter_space,
                          seed=seed, R=R, eta=eta, n_jobs=n_jobs)

    def iterate(self, num_iter=1):
        '''
            Iterate a SH procedure (inner loop) in Hyperband.
        :return:
        '''
        _start_time = time.time()
        for _ in range(num_iter):
            self._mfse_iterate(self.s_values[self.inner_iter_id])
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.s_max + 1)

        iteration_cost = time.time() - _start_time
        inc_idx = np.argmin(np.array(self.incumbent_perfs))

        for idx in range(len(self.incumbent_perfs)):
            self.eval_dict[(self.incumbent_configs[idx], self.evaluator.hpo_config)] = -self.incumbent_perfs[idx]
        self.incumbent_perf = -self.incumbent_perfs[inc_idx]
        self.incumbent_config = self.incumbent_configs[inc_idx]
        # incumbent_perf: the large the better
        return self.incumbent_perf, iteration_cost, self._parse(self.root_node, self.incumbent_config)

    def fetch_nodes(self, n=5):
        hist_dict = dict()
        for key, value in self.eval_dict.items():
            hist_dict[key[0]] = value

        max_list = sorted(hist_dict.items(), key=lambda item: item[1], reverse=True)

        if len(max_list) < 50:
            max_n = list(max_list[:n])
        else:
            amplification_rate = 3
            chosen_idxs = np.arange(n) * amplification_rate
            max_n = [max_list[idx] for idx in chosen_idxs]

        if self.incumbent_config not in [x[0] for x in max_n]:
            max_n.append((self.incumbent_config, hist_dict[self.incumbent_config]))

        node_list = []
        for i, config in enumerate(max_n):
            if config[0] in self.node_dict:
                node_list.append(self.node_dict[config[0]][0])
                continue

            try:
                node, tran_list = self._parse(self.root_node, config[0], record=True)
                node.config = config[0]
                if node.data[0].shape[1] == 0:
                    continue
                if self.fetch_incumbent is None:
                    self.fetch_incumbent = node  # Update incumbent node
                node_list.append(node)
                self.node_dict[config[0]] = [node, tran_list]
            except:
                print("Re-parse failed on config %s" % str(config[0]))
        return node_list
