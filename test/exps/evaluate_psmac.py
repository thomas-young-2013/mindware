import multiprocessing
import os
import sys
import argparse
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
import time
import datetime
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.optimizer import pSMAC

sys.path.append(os.getcwd())
from automlToolkit.components.hpo_optimizer.base_optimizer import BaseHPOptimizer
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.datasets.utils import load_data
from automlToolkit.components.evaluator import Evaluator

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--dataset', type=str, default='diabetes')
parser.add_argument('--optimizer', type=str, default='smac', choices=['smac', 'psmac'])
parser.add_argument('--n', type=int, default=4)
parser.add_argument('--algo', type=str, default='extra_trees')
parser.add_argument('--runcount_limit', type=int, default=50)
parser.add_argument('--seed', type=int, default=1)


class PSMACOptimizer(BaseHPOptimizer):
    def __init__(self, evaluator, config_space, n_jobs=4, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, output_dir='./', trials_per_iter=1, seed=1):
        super().__init__(evaluator, config_space, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.n_jobs = n_jobs

        if not output_dir.endswith('/'):
            output_dir += '/'
        self.output_dir = output_dir
        output_dir += "psmac3_output_%s" % (datetime.datetime.fromtimestamp(
            time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))
        self.scenario_dict = {'abort_on_first_run_crash': False,
                              "run_obj": "quality",
                              "cs": self.config_space,
                              "deterministic": "true",
                              "shared-model": True,  # PSMAC Entry
                              "runcount-limit": self.evaluation_num_limit,
                              "output_dir": output_dir,
                              "cutoff_time": self.per_run_time_limit
                              }
        self.optimizer_list = []
        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = -1.
        self.incumbent_config = self.config_space.get_default_configuration()
        # Estimate the size of the hyperparameter space.
        hp_num = len(self.config_space.get_hyperparameters())
        if hp_num == 0:
            self.config_num_threshold = 0
        else:
            _threshold = int(len(set(self.config_space.sample_configuration(12500))) * 0.8)
            self.config_num_threshold = _threshold
        self.logger.info('HP_THRESHOLD is: %d' % self.config_num_threshold)

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

        for i in range(self.n_jobs):
            runhistory = return_hist[i]
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


def conduct_hpo(optimizer='smac', dataset='pc4', classifier_id='random_forest', runcount_limit=100):
    from autosklearn.pipeline.components.classification import _classifiers

    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)

    raw_data = load_data(dataset, datanode_returned=True)
    print(set(raw_data.data[1]))
    evaluator = Evaluator(cs.get_default_configuration(), name='hpo', data_node=raw_data)

    if optimizer == 'smac':
        optimizer = SMACOptimizer(evaluator, cs, evaluation_limit=runcount_limit, output_dir='logs')
    elif optimizer == 'psmac':
        optimizer = PSMACOptimizer(evaluator, cs, args.n, evaluation_limit=runcount_limit, output_dir='logs')
    inc, val = optimizer.optimize()
    print(inc, val)


if __name__ == "__main__":
    args = parser.parse_args()
    conduct_hpo(optimizer=args.optimizer, dataset=args.dataset, classifier_id=args.algo,
                runcount_limit=args.runcount_limit)
