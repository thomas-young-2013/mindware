import time
import numpy as np
from math import log, ceil

from automlToolkit.components.hpo_optimizer.base_optimizer import BaseHPOptimizer
from automlToolkit.components.hpo_optimizer.utils.prob_rf_cluster import WeightedRandomForestCluster
from automlToolkit.components.hpo_optimizer.utils.funcs import get_types, minmax_normalization
from automlToolkit.components.hpo_optimizer.utils.config_space_utils import convert_configurations_to_array


class MfseOptimizer(BaseHPOptimizer):
    def __init__(self, evaluator, config_space, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, per_run_mem_limit=1024, output_dir='./', trials_per_iter=1, seed=1,
                 R=81, eta=3):
        super().__init__(evaluator, config_space, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.config_space = config_space

        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = -1.
        self.incumbent_config = self.config_space.get_default_configuration()
        self.incumbent_configs = []
        self.incumbent_perfs = []

        # Parameters in Hyperband framework.
        self.restart_needed = True
        self.R = R
        self.eta = eta
        self.seed = seed
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.s_values = list(reversed(range(self.s_max + 1)))
        self.inner_iter_id = 0

        # Parameters in MFSE-HB.
        self.weight_update_id = 0
        self.iterate_r = []
        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = []
            self.target_y[r] = []

        types, bounds = get_types(config_space)
        self.num_config = len(bounds)
        init_weight = [0.]
        init_weight.extend([1. / self.s_max] * self.s_max)
        self.weighted_surrogate = WeightedRandomForestCluster(types, bounds, self.s_max,
                                                              self.eta, init_weight, 'gpoe')

        # TODO: need to improve with lite-bo.
        # self.weighted_acquisition_func = EI(model=self.weighted_surrogate)
        # self.weighted_acq_optimizer = RandomSampling(self.weighted_acquisition_func,
        #                                              config_space, n_samples=max(500, 50 * self.num_config))

    def iterate(self, num_iter=1):
        '''
            Iterate a SH procedure (inner loop) in Hyperband.
        :return:
        '''
        _start_time = time.time()
        for _ in range(num_iter):
            self._iterate(self.s_values[self.inner_iter_id])
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.s_max + 1)

        iteration_cost = time.time() - _start_time
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def _iterate(self, s, skip_last=0):

        # TODO: update the model weight.
        if self.weight_update_id > self.s_max:
            self.update_weight()
        self.weight_update_id += 1

        # Set initial number of configurations
        n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
        # initial number of iterations per config
        r = int(self.R * self.eta ** (-s))

        # Choose a batch of configurations in different mechanisms.
        start_time = time.time()
        # TODO: sample candidate configurations.
        T = self.fetch_candidate_configurations(n)
        time_elapsed = time.time() - start_time
        self.logger.info("Choosing next configurations took %.2f sec." % time_elapsed)

        extra_info = None
        last_run_num = None

        for i in range((s + 1) - int(skip_last)):  # changed from s + 1

            # Run each of the n configs for <iterations>
            # and keep best (n_configs / eta) configurations

            n_configs = n * self.eta ** (-i)
            n_iterations = r * self.eta ** (i)

            n_iter = n_iterations
            if last_run_num is not None and not self.restart_needed:
                n_iter -= last_run_num
            last_run_num = n_iterations

            self.logger.info("MFSE: %d configurations x %d iterations each" %
                             (int(n_configs), int(n_iterations)))

            ret_val, early_stops = self.run_in_parallel(T, n_iter, extra_info)
            val_losses = [item['loss'] for item in ret_val]
            ref_list = [item['ref_id'] for item in ret_val]

            self.target_x[int(n_iterations)].extend(T)
            self.target_y[int(n_iterations)].extend(val_losses)

            if int(n_iterations) == self.R:
                self.incumbent_configs.extend(T)
                self.incumbent_perfs.extend(val_losses)

            # Select a number of best configurations for the next loop.
            # Filter out early stops, if any.
            indices = np.argsort(val_losses)
            if len(T) == sum(early_stops):
                break
            if len(T) >= self.eta:
                T = [T[i] for i in indices if not early_stops[i]]
                extra_info = [ref_list[i] for i in indices if not early_stops[i]]
                reduced_num = int(n_configs / self.eta)
                T = T[0:reduced_num]
                extra_info = extra_info[0:reduced_num]
            else:
                T = [T[indices[0]]]
                extra_info = [ref_list[indices[0]]]
            incumbent_loss = val_losses[indices[0]]

            for item in self.iterate_r[self.iterate_r.index(r):]:
                # NORMALIZE Objective value: MinMax linear normalization
                normalized_y = minmax_normalization(self.target_y[item])
                self.weighted_surrogate.train(convert_configurations_to_array(self.target_x[item]),
                                              np.array(normalized_y, dtype=np.float64), r=item)

    def fetch_candidate_configurations(self, n):
        pass

    def update_weight(self):
        pass
