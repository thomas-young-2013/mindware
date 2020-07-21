import time
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

from solnml.utils.logging_utils import get_logger
from solnml.components.utils.constants import IMG_CLS, TEXT_CLS, OBJECT_DET
from solnml.components.hpo_optimizer.base.config_space_utils import sample_configurations
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space
from solnml.components.computation.parallel_process import ParallelProcessEvaluator


class CashpOptimizer(object):
    def __init__(self, task_type, architectures, time_limit,
                 R=27, eta=3, N=9, n_jobs=1):
        self.architectures = architectures
        self.time_limit = time_limit
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.R = R
        self.eta = eta
        self.N = N
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

    def sample_configs_for_archs(self, include_architectures, N):
        configs = list()
        for _arch in include_architectures:
            _cs = self.get_model_config_space(_arch)
            configs.extend(sample_configurations(_cs, N))
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
            while True:
                r = 1
                C = self.sample_configs_for_archs(architecture_candidates, self.N)
                while r < self.R:
                    for _arch in architecture_candidates:
                        if r not in self.eval_hist_configs[_arch]:
                            self.eval_hist_configs[_arch][r] = list()
                            self.eval_hist_perfs[_arch][r] = list()

                    self.logger.info('Start to evaluate %d configurations with %d resource' % (len(C), r))
                    _start_time = time.time()
                    if _start_time >= start_time + self.time_limit:
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
                    self.logger.info('Evaluation took %.2f seconds' % (time.time() - _start_time))

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
                            val_losses_previous_iter = self.query_performance(C, r//self.eta)
                            previous_inc_loss = np.min(val_losses_previous_iter)
                            indices = np.argsort(val_losses)
                            C = [C[idx] for idx in indices if val_losses[idx] < previous_inc_loss]
                    r *= self.eta
                    if inc_perf > val_losses[indices[0]]:
                        inc_perf = val_losses[indices[0]]
                        inc_config = C[0]

                archs, reduced_archs = [config['estimator'] for config in C], list()
                # Preserve the partial-relationship order.
                for _arch in archs:
                    if _arch not in reduced_archs:
                        reduced_archs.append(_arch)

                architecture_candidates = reduced_archs
                print('Reduced architectures:', architecture_candidates)
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
