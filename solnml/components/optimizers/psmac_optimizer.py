import multiprocessing
import time
import datetime
import numpy as np
import os
import pickle
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.optimizer import pSMAC

from solnml.components.optimizers.base_optimizer import BaseOptimizer


class PSMACOptimizer(BaseOptimizer):
    def __init__(self, evaluator, config_space, name, n_jobs=4, time_limit=None, evaluation_limit=200,
                 per_run_time_limit=600, per_run_mem_limit=1024, output_dir='./', trials_per_iter=1, seed=1):
        super().__init__(evaluator, config_space, name, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.trials_this_run = trials_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.n_jobs = n_jobs

        if not output_dir.endswith('/'):
            output_dir += '/'
        self.output_dir = output_dir
        output_dir += "psmac3_output_%s" % (datetime.datetime.fromtimestamp(
            time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))
        self.output_dir = output_dir
        self.scenario_dict = {'abort_on_first_run_crash': False,
                              "run_obj": "quality",
                              "cs": self.config_space,
                              "deterministic": "true",
                              "shared-model": True,  # PSMAC Entry
                              "runcount-limit": self.evaluation_num_limit,
                              "output_dir": output_dir,
                              "cutoff_time": self.per_run_time_limit
                              }
        self.optimizer_list = list()
        for _ in range(self.n_jobs):
            self.optimizer_list.append(SMAC(scenario=Scenario(self.scenario_dict),
                                            rng=np.random.RandomState(None),  # Different seed for different optimizers
                                            tae_runner=self.evaluator))
        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = float("-INF")
        self.incumbent_config = self.config_space.get_default_configuration()
        # Estimate the size of the hyperparameter space.
        hp_num = len(self.config_space.get_hyperparameters())
        if hp_num == 0:
            self.config_num_threshold = 0
        else:
            _threshold = int(len(set(self.config_space.sample_configuration(12500))) * 0.8)
            self.config_num_threshold = _threshold
        self.logger.info('HP_THRESHOLD is: %d' % self.config_num_threshold)

    def run(self):
        while True:
            if self.evaluation_num_limit is not None and self.trial_cnt > self.evaluation_num_limit:
                break
            if self.time_limit is not None and time.time() - self.start_time > self.time_limit:
                break
            if self.evaluation_num_limit - self.trial_cnt > self.trials_per_iter:
                self.trials_this_run = self.trials_per_iter
            else:
                self.trials_this_run = self.evaluation_num_limit - self.trial_cnt
            self.iterate()

        return np.max(self.perfs)

    def iterate(self):
        trial_left = multiprocessing.Value('i', self.trials_this_run)
        _start_time = time.time()
        _flag = False
        if len(self.configs) >= self.config_num_threshold:
            _flag = True
            self.logger.warning('Already explored 70 percentage of the '
                                'hp space: %d!' % self.config_num_threshold)
        else:
            # for i in range(self.n_jobs):
            #     self.trial_statistics.append(self.pool.submit(_iterate,
            #                                                   self.optimizer_list[i], trial_left))
            # self.wait_tasks_finish()
            processes = []
            return_hist = multiprocessing.Manager().list()
            for i in range(self.n_jobs):
                pSMAC.read(
                    run_history=self.optimizer_list[i].solver.runhistory,
                    output_dirs=self.optimizer_list[i].solver.scenario.output_dir + '/run_1',
                    configuration_space=self.optimizer_list[i].solver.config_space,
                    logger=self.optimizer_list[i].solver.logger,
                )
            for i in range(self.n_jobs):
                p = multiprocessing.Process(
                    target=_iterate,
                    args=[self.optimizer_list[i], trial_left, return_hist]
                )
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            for runhistory in return_hist:
                runkeys = list(runhistory.data.keys())
                for key in runkeys:
                    _reward = 1. - runhistory.data[key][0]
                    _config = runhistory.ids_config[key[0]]
                    if _config not in self.configs:
                        self.perfs.append(_reward)
                        self.configs.append(_config)
                    if _reward > self.incumbent_perf:
                        self.incumbent_perf = _reward
                        self.incumbent_config = _config
            self.trial_cnt += self.trials_per_iter
        if not _flag:
            iteration_cost = time.time() - _start_time
        else:
            iteration_cost = None
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def optimize(self):
        for i in range(self.n_jobs):
            self.optimizer_list.append(SMAC(scenario=Scenario(self.scenario_dict),
                                            rng=np.random.RandomState(None),  # Different seed for different optimizers
                                            tae_runner=self.evaluator))

        processes = []
        return_hist = multiprocessing.Manager().list()
        for i in range(1, self.n_jobs):
            p = multiprocessing.Process(
                target=self._optimize,
                args=[self.optimizer_list[i], return_hist]
            )
            processes.append(p)
            p.start()
        self._optimize(self.optimizer_list[0], return_hist)
        for p in processes:
            p.join()

        for runhistory in return_hist:
            runkeys = list(runhistory.data.keys())
            for key in runkeys:
                _reward = 1. - runhistory.data[key][0]
                _config = runhistory.ids_config[key[0]]
                if _config not in self.configs:
                    self.perfs.append(_reward)
                    self.configs.append(_config)
                if _reward > self.incumbent_perf:
                    self.incumbent_perf = _reward
                    self.incumbent_config = _config
        return self.incumbent_config, self.incumbent_perf

    def _optimize(self, optimizer, hist_list):
        optimizer.optimize()
        pSMAC.read(
            run_history=optimizer.solver.runhistory,
            output_dirs=optimizer.solver.scenario.input_psmac_dirs,
            configuration_space=optimizer.solver.config_space,
            logger=optimizer.solver.logger,
        )
        hist_list.append(optimizer.solver.runhistory)

    def wait_tasks_finish(self):
        all_completed = False
        while not all_completed:
            all_completed = True
            for trial in self.trial_statistics:
                if not trial.done():
                    all_completed = False
                    time.sleep(0.1)
                    break


def _iterate(optimizer, runcount_left, return_hist):
    while runcount_left.value > 0:
        runcount_left.value -= 1
        optimizer.iterate()
    pSMAC.read(
        run_history=optimizer.solver.runhistory,
        output_dirs=optimizer.solver.scenario.input_psmac_dirs,
        configuration_space=optimizer.solver.config_space,
        logger=optimizer.solver.logger,
    )
    # print(optimizer.solver.runhistory.data)
    return_hist.append(optimizer.solver.runhistory)
