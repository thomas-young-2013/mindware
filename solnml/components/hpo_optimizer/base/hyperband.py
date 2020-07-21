import time
import numpy as np
from math import log, ceil

from solnml.utils.constant import MAX_INT
from solnml.utils.logging_utils import get_logger
from solnml.components.hpo_optimizer.base.config_space_utils import sample_configurations
from solnml.components.computation.parallel_process import ParallelProcessEvaluator


class HyperbandBase(object):
    def __init__(self, eval_func, config_space,
                 seed=1, R=81, eta=3, n_jobs=1):
        self.eval_func = eval_func
        self.config_space = config_space
        self.n_workers = n_jobs

        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = float("-INF")
        self.incumbent_config = self.config_space.get_default_configuration()
        self.incumbent_configs = []
        self.incumbent_perfs = []
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

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

        # Parameters in Hyperband.
        self.iterate_r = list()
        self.target_x = dict()
        self.target_y = dict()
        self.exp_output = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = list()
            self.target_y[r] = list()

        self.eval_dict = dict()

    def _iterate(self, s, budget=MAX_INT, skip_last=0):

        # Set initial number of configurations
        n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
        # initial number of iterations per config
        r = int(self.R * self.eta ** (-s))

        # Choose a batch of configurations in different mechanisms.
        start_time = time.time()
        T = sample_configurations(self.config_space, n)
        time_elapsed = time.time() - start_time
        self.logger.info("Choosing next batch of configurations took %.2f sec." % time_elapsed)

        with ParallelProcessEvaluator(self.eval_func, n_worker=self.n_workers) as executor:
            for i in range((s + 1) - int(skip_last)):  # changed from s + 1
                if time.time() >= budget + start_time:
                    break

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_resource = r * self.eta ** i

                self.logger.info("MFSE: %d configurations x size %d / %d each" %
                                 (int(n_configs), n_resource, self.R))

                val_losses = executor.parallel_execute(T, resource_ratio=float(n_resource / self.R),
                                                       eta=self.eta,
                                                       first_iter=(i == 0))
                for _id, _val_loss in enumerate(val_losses):
                    if np.isfinite(_val_loss):
                        self.target_x[int(n_resource)].append(T[_id])
                        self.target_y[int(n_resource)].append(_val_loss)

                self.exp_output[time.time()] = (int(n_resource), T, val_losses)

                if int(n_resource) == self.R:
                    self.incumbent_configs.extend(T)
                    self.incumbent_perfs.extend(val_losses)

                # Select a number of best configurations for the next loop.
                # Filter out early stops, if any.
                indices = np.argsort(val_losses)
                if len(T) >= self.eta:
                    T = [T[i] for i in indices]
                    reduced_num = int(n_configs / self.eta)
                    T = T[0:reduced_num]
                else:
                    T = [T[indices[0]]]
