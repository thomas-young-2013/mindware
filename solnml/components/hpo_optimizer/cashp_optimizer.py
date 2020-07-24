import time
import os
import random as rd
import numpy as np
from math import log
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

from solnml.utils.logging_utils import get_logger
from solnml.components.utils.constants import IMG_CLS, TEXT_CLS, OBJECT_DET
from solnml.components.hpo_optimizer.base.config_space_utils import sample_configurations
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space
from solnml.components.computation.parallel_process import ParallelProcessEvaluator
from solnml.components.transfer_learning.tlbo.models.kde import TPE
from solnml.components.hpo_optimizer.base.acquisition import EI
from solnml.components.hpo_optimizer.base.acq_optimizer import RandomSampling
from solnml.components.hpo_optimizer.base.prob_rf_cluster import WeightedRandomForestCluster
from solnml.components.hpo_optimizer.base.funcs import get_types, std_normalization
from solnml.components.hpo_optimizer.base.config_space_utils import convert_configurations_to_array


class CashpOptimizer(object):
    def __init__(self, task_type, architectures, time_limit, sampling_strategy='uniform',
                 R=27, eta=3, N=9, n_jobs=1):
        self.architectures = architectures
        self.time_limit = time_limit
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.R = R
        self.eta = eta
        self.N = N
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.sampling_strategy = sampling_strategy
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        from solnml.components.models.img_classification import _classifiers as _img_estimators, _addons as _img_addons
        from solnml.components.models.text_classification import _classifiers as _text_estimators, \
            _addons as _text_addons
        from solnml.components.models.object_detection import _classifiers as _od_estimators, _addons as _od_addons

        self.time_limit = time_limit
        self.elimination_strategy = 'bandit'
        # Runtime stats.
        self.evaluation_stats = dict()

        self.update_cs = dict()

        if task_type == IMG_CLS:
            self._estimators = _img_estimators
            self._addons = _img_addons
        elif task_type == TEXT_CLS:
            self._estimators = _text_estimators
            self._addons = _text_addons
        elif task_type == OBJECT_DET:
            self._estimators = _od_estimators
            self._addons = _od_addons
        else:
            raise ValueError("Unknown task type %s" % task_type)
        self.eval_hist_configs = dict()
        self.eval_hist_perfs = dict()

        self.tpe_config_gen = dict()
        self.mfse_config_gen = dict()

    def get_model_config_space(self, estimator_id, include_estimator=True, include_aug=True):
        if estimator_id in self._estimators:
            clf_class = self._estimators[estimator_id]
        elif estimator_id in self._addons.components:
            clf_class = self._addons.components[estimator_id]
        else:
            raise ValueError("Algorithm %s not supported!" % estimator_id)

        default_cs = clf_class.get_hyperparameter_search_space()
        model = UnParametrizedHyperparameter("estimator", estimator_id)
        if include_estimator:
            default_cs.add_hyperparameter(model)
        if self.task_type == IMG_CLS and include_aug is True:
            aug_space = get_aug_hyperparameter_space()
            default_cs.add_hyperparameters(aug_space.get_hyperparameters())
            default_cs.add_conditions(aug_space.get_conditions())

        # Update configuration space according to config file
        all_cs = self.update_cs.get('all', ConfigurationSpace())
        all_hp_names = all_cs.get_hyperparameter_names()
        estimator_cs = self.update_cs.get(estimator_id, ConfigurationSpace())
        estimator_hp_names = estimator_cs.get_hyperparameter_names()

        cs = ConfigurationSpace()
        for hp_name in default_cs.get_hyperparameter_names():
            if hp_name in estimator_hp_names:
                cs.add_hyperparameter(estimator_cs.get_hyperparameter(hp_name))
            elif hp_name in all_hp_names:
                cs.add_hyperparameter(all_cs.get_hyperparameter(hp_name))
            else:
                cs.add_hyperparameter(default_cs.get_hyperparameter(hp_name))
        return cs

    def sample_configs_for_archs(self, include_architectures, N, sampling_strategy='uniform'):
        configs = list()
        for _arch in include_architectures:
            _cs = self.get_model_config_space(_arch)
            if sampling_strategy == 'uniform':
                configs.extend(sample_configurations(_cs, N))

            elif sampling_strategy == 'bohb':
                if _arch not in self.tpe_config_gen:
                    self.tpe_config_gen[_arch] = TPE(_cs)
                config_candidates = list()

                config_left = N
                while config_left:
                    config = self.tpe_config_gen[_arch].get_config()[0]
                    if config in config_candidates:
                        continue
                    config_candidates.append(config)
                    config_left -= 1

                p_threshold = 0.3
                idx_acq = 0
                for _id in range(N):
                    if rd.random() < p_threshold:
                        config = sample_configurations(_cs, 1)[0]
                    else:
                        config = config_candidates[idx_acq]
                        idx_acq += 1
                    configs.append(config)

            else:  # mfse
                if _arch not in self.mfse_config_gen:
                    types, bounds = get_types(_cs)
                    init_weight = [1. / self.s_max] * self.s_max + [0.]
                    self.mfse_config_gen[_arch] = dict()
                    self.mfse_config_gen[_arch]['surrogate'] = WeightedRandomForestCluster(types, bounds, self.s_max,
                                                                                           self.eta, init_weight,
                                                                                           'gpoe')
                    acq_func = EI(model=self.mfse_config_gen[_arch]['surrogate'])
                    self.mfse_config_gen[_arch]['acq_optimizer'] = RandomSampling(acq_func, _cs,
                                                                                  n_samples=2000,
                                                                                  rng=np.random.RandomState(1))
                if self.R not in self.eval_hist_perfs[_arch] or len(self.eval_hist_perfs[_arch][self.R]) == 0:
                    configs.extend(sample_configurations(_cs, N))
                    continue

                incumbent = dict()
                max_r = self.R
                # The lower, the better.
                best_index = np.argmin(self.eval_hist_perfs[_arch][max_r])
                incumbent['config'] = self.eval_hist_configs[_arch][max_r][best_index]
                approximate_obj = self.mfse_config_gen[_arch]['surrogate'].predict(
                    convert_configurations_to_array([incumbent['config']]))[0]
                incumbent['obj'] = approximate_obj
                self.mfse_config_gen[_arch]['acq_optimizer'].update(model=self.mfse_config_gen[_arch]['surrogate'],
                                                                    eta=incumbent)

                config_candidates = self.mfse_config_gen[_arch]['acq_optimizer'].maximize(batch_size=N)
                p_threshold = 0.3
                n_acq = self.eta * self.eta

                if N <= n_acq:
                    return config_candidates

                candidates = config_candidates[: n_acq]
                idx_acq = n_acq
                for _id in range(N - n_acq):
                    if rd.random() < p_threshold:
                        config = sample_configurations(_cs, 1)[0]
                    else:
                        config = config_candidates[idx_acq]
                        idx_acq += 1
                    candidates.append(config)
                return candidates
        return configs

    """
        iteration procedure.
        iter_1: r=1, n=5*9
        iter_2: r=3, n=5*3
        iter_3: r=9, n=5
    """

    def run(self, dl_evaluator):
        start_time = time.time()
        inc_config, inc_perf = None, np.inf
        architecture_candidates = self.architectures.copy()
        for _arch in architecture_candidates:
            self.eval_hist_configs[_arch] = dict()
            self.eval_hist_perfs[_arch] = dict()
        self.evaluation_stats['timestamps'] = list()
        self.evaluation_stats['val_scores'] = list()

        with ParallelProcessEvaluator(dl_evaluator, n_worker=self.n_jobs) as executor:
            terminate_proc = False
            while not terminate_proc:
                r = 1
                C = self.sample_configs_for_archs(architecture_candidates, self.N,
                                                  sampling_strategy=self.sampling_strategy)
                while r < self.R or (r == self.R and len(architecture_candidates) == 1):
                    for _arch in architecture_candidates:
                        if r not in self.eval_hist_configs[_arch]:
                            self.eval_hist_configs[_arch][r] = list()
                            self.eval_hist_perfs[_arch][r] = list()

                    self.logger.info('Evalutions [r=%d]' % r)
                    self.logger.info('Start to evaluate %d configurations with %d resource' % (len(C), r))
                    self.logger.info('=' * 20)
                    _start_time = time.time()
                    if _start_time >= start_time + self.time_limit:
                        terminate_proc = True
                        break

                    if self.n_jobs > 1:
                        val_losses = executor.parallel_execute(C, resource_ratio=float(r / self.R),
                                                               eta=self.eta, first_iter=(r == 1))
                        for _id, val_loss in enumerate(val_losses):
                            if np.isfinite(val_loss):
                                _arch = C[_id]['estimator']
                                self.eval_hist_configs[_arch][r].append(C[_id])
                                self.eval_hist_perfs[_arch][r].append(val_loss)
                                self.evaluation_stats['timestamps'].append(time.time() - start_time)
                                self.evaluation_stats['val_scores'].append(val_loss)
                    else:
                        val_losses = list()
                        for config in C:
                            val_loss = dl_evaluator(config, resource_ratio=float(r / self.R),
                                                    eta=self.eta, first_iter=(r == 1))
                            val_losses.append(val_loss)
                            if np.isfinite(val_loss):
                                _arch = config['estimator']
                                self.eval_hist_configs[_arch][r].append(config)
                                self.eval_hist_perfs[_arch][r].append(val_loss)
                                self.evaluation_stats['timestamps'].append(time.time() - start_time)
                                self.evaluation_stats['val_scores'].append(val_loss)
                    self.logger.info('Evaluations [R=%d] took %.2f seconds' % (r, time.time() - _start_time))

                    # Train surrogate
                    if self.sampling_strategy == 'bohb':
                        if r == self.R:
                            for i, _config in enumerate(C):
                                if np.isfinite(val_losses[i]):
                                    _arch = _config['estimator']
                                    self.tpe_config_gen[_arch].new_result(_config, val_losses[i], r)
                    elif self.sampling_strategy == 'mfse':
                        for _arch in architecture_candidates:  # Only update surrogate in candidates
                            normalized_y = std_normalization(self.eval_hist_perfs[_arch][r])
                            if len(self.eval_hist_configs[_arch][r]) == 0:  # No configs for this architecture
                                continue
                            self.mfse_config_gen[_arch]['surrogate'].train(
                                convert_configurations_to_array(self.eval_hist_configs[_arch][r]),
                                np.array(normalized_y, dtype=np.float64), r=r)

                    if self.elimination_strategy == 'bandit':
                        indices = np.argsort(val_losses)
                        if len(C) >= self.eta:
                            C = [C[i] for i in indices]
                            reduced_num = int(len(C) / self.eta)
                            C = C[0:reduced_num]
                        else:
                            C = [C[indices[0]]]

                    else:
                        if r > 1:
                            val_losses_previous_iter = self.query_performance(C, r // self.eta)
                            previous_inc_loss = np.min(val_losses_previous_iter)
                            indices = np.argsort(val_losses)
                            C = [C[idx] for idx in indices if val_losses[idx] < previous_inc_loss]

                    if inc_perf > val_losses[indices[0]]:
                        inc_perf = val_losses[indices[0]]
                        inc_config = C[0]
                    r *= self.eta

                # Remove tmp model
                if dl_evaluator.continue_training:
                    for filename in os.listdir(dl_evaluator.model_dir):
                        # Temporary model
                        if 'tmp_%s' % dl_evaluator.timestamp in filename:
                            try:
                                filepath = os.path.join(dl_evaluator.model_dir, filename)
                                os.remove(filepath)
                            except Exception:
                                pass

                archs, reduced_archs = [config['estimator'] for config in C], list()
                # Preserve the partial-relationship order.
                for _arch in archs:
                    if _arch not in reduced_archs:
                        reduced_archs.append(_arch)

                architecture_candidates = reduced_archs
                print('=' * 20)
                print('Reduced architectures:', architecture_candidates)
                print('=' * 20)
        return inc_config, inc_perf

    def query_performance(self, C, r):
        perfs = list()
        for config in C:
            arch = config['estimator']
            idx = self.eval_hist_configs[arch][r].index(config)
            perfs.append(self.eval_hist_perfs[arch][r][idx])
        return perfs

    def get_evaluation_stats(self):
        return self.evaluation_stats
