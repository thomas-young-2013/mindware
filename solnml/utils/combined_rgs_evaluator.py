from ConfigSpace import ConfigurationSpace, UnParametrizedHyperparameter, CategoricalHyperparameter
import os
import time
import warnings
import numpy as np
import pickle as pkl
from multiprocessing import Lock
from sklearn.metrics.scorer import balanced_accuracy_scorer

from solnml.utils.logging_utils import get_logger
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.evaluators.evaluate_func import validation
from solnml.components.fe_optimizers.task_space import get_task_hyperparameter_space
from solnml.components.fe_optimizers.parse import parse_config, construct_node
from solnml.components.evaluators.base_evaluator import CombinedTopKModelSaver
from solnml.components.utils.class_loader import get_combined_candidtates
from solnml.components.models.regression import _regressors, _addons
from solnml.components.utils.constants import *


def get_estimator(config, estimator_id):
    regressor_type = estimator_id
    config_ = config.copy()
    config_['%s:random_state' % regressor_type] = 1
    hpo_config = dict()
    for key in config_:
        key_name = key.split(':')[0]
        if regressor_type == key_name:
            act_key = key.split(':')[1]
            hpo_config[act_key] = config_[key]

    _candidates = get_combined_candidtates(_regressors, _addons)
    estimator = _candidates[regressor_type](**hpo_config)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', 1)
    return regressor_type, estimator


def get_hpo_cs(estimator_id, task_type=REGRESSION):
    _candidates = get_combined_candidtates(_regressors, _addons)
    if estimator_id in _candidates:
        rgs_class = _candidates[estimator_id]
    else:
        raise ValueError("Algorithm %s not supported!" % estimator_id)
    cs = rgs_class.get_hyperparameter_search_space()
    return cs


def get_fe_cs(estimator_id, task_type=REGRESSION, include_image=False, include_text=False, include_preprocessors=None):
    cs = get_task_hyperparameter_space(task_type=task_type, estimator_id=estimator_id,
                                       include_image=include_image, include_text=include_text,
                                       include_preprocessors=include_preprocessors)
    return cs


def get_combined_cs(estimator_id, task_type=REGRESSION, include_image=False, include_text=False,
                    include_preprocessors=None, if_imbal=False):
    cs = ConfigurationSpace()
    hpo_cs = get_hpo_cs(estimator_id, task_type)
    fe_cs = get_fe_cs(estimator_id, task_type,
                      include_image=include_image, include_text=include_text,
                      include_preprocessors=include_preprocessors)
    config_cand = [estimator_id]
    config_option = CategoricalHyperparameter('hpo', config_cand)
    cs.add_hyperparameter(config_option)
    for config_item in config_cand:
        sub_configuration_space = hpo_cs
        parent_hyperparameter = {'parent': config_option,
                                 'value': config_item}
        cs.add_configuration_space(config_item, sub_configuration_space,
                                   parent_hyperparameter=parent_hyperparameter)
    for hp in fe_cs.get_hyperparameters():
        cs.add_hyperparameter(hp)
    for cond in fe_cs.get_conditions():
        cs.add_condition(cond)
    for bid in fe_cs.get_forbiddens():
        cs.add_forbidden_clause(bid)
    return cs


class CombinedRegressionEvaluator(_BaseEvaluator):
    def __init__(self, estimator_id, scorer=None, data_node=None, task_type=REGRESSION, resampling_strategy='cv',
                 resampling_params=None, timestamp=None, output_dir=None, seed=1, if_imbal=False):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params

        self.estimator_id = estimator_id
        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.task_type = task_type
        self.data_node = data_node
        self.output_dir = output_dir
        self.seed = seed
        self.onehot_encoder = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.continue_training = False

        self.train_node = data_node.copy_()
        self.val_node = data_node.copy_()

        self.timestamp = timestamp
        # TODO: Top-k k?
        self.topk_model_saver = CombinedTopKModelSaver(k=60, model_dir=self.output_dir, identifier=timestamp)

    def __call__(self, config, **kwargs):
        start_time = time.time()
        return_dict = dict()
        self.seed = 1
        downsample_ratio = kwargs.get('resource_ratio', 1.0)

        if 'holdout' in self.resampling_strategy:
            try:
                # Prepare data node.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    if self.resampling_params is None or 'test_size' not in self.resampling_params:
                        test_size = 0.33
                    else:
                        test_size = self.resampling_params['test_size']

                    from sklearn.model_selection import ShuffleSplit
                    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                    for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                        _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                        _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                    self.train_node.data = [_x_train, _y_train]
                    self.val_node.data = [_x_val, _y_val]

                    data_node, op_list = parse_config(self.train_node, config, record=True)
                    _val_node = self.val_node.copy_()
                    _val_node = construct_node(_val_node, op_list)

                _x_train, _y_train = data_node.data
                _x_val, _y_val = _val_node.data

                config_dict = config.get_dictionary().copy()
                # regression gadgets
                regressor_id, clf = get_estimator(config_dict, self.estimator_id)

                score = validation(clf, self.scorer, _x_train, _y_train, _x_val, _y_val,
                                   random_state=self.seed)

                if 'rw_lock' not in kwargs or kwargs['rw_lock'] is None:
                    self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
                lock = kwargs.get('rw_lock', Lock())
                lock.acquire()
                if np.isfinite(score):
                    save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(config, score,
                                                                                                       regressor_id)
                    if save_flag is True:
                        with open(model_path, 'wb') as f:
                            pkl.dump([op_list, clf], f)
                        self.logger.info("Model saved to %s" % model_path)

                    try:
                        if delete_flag and os.path.exists(model_path_deleted):
                            os.remove(model_path_deleted)
                            self.logger.info("Model deleted from %s" % model_path)
                    except:
                        pass
                lock.release()

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.logger.info('Evaluator: %s' % (str(e)))
                score = -np.inf

        elif 'cv' in self.resampling_strategy:
            # Prepare data node.
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    if 'cv' in self.resampling_strategy:
                        if self.resampling_params is None or 'folds' not in self.resampling_params:
                            folds = 5
                        else:
                            folds = self.resampling_params['folds']

                    from sklearn.model_selection import KFold
                    kfold = KFold(n_splits=folds, random_state=self.seed, shuffle=False)
                    scores = list()

                    for train_index, test_index in kfold.split(self.data_node.data[0], self.data_node.data[1]):
                        _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                        _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                        self.train_node.data = [_x_train, _y_train]
                        self.val_node.data = [_x_val, _y_val]

                        data_node, op_list = parse_config(self.train_node, config, record=True)
                        _val_node = self.val_node.copy_()
                        _val_node = construct_node(_val_node, op_list)

                        _x_train, _y_train = data_node.data
                        _x_val, _y_val = _val_node.data

                        config_dict = config.get_dictionary().copy()
                        # regressor gadgets
                        regressor_id, clf = get_estimator(config_dict, self.estimator_id)

                        _score = validation(clf, self.scorer, _x_train, _y_train, _x_val, _y_val,
                                            random_state=self.seed)
                        scores.append(_score)
                    score = np.mean(scores)

                # TODO: Don't save models for cv
                if 'rw_lock' not in kwargs or kwargs['rw_lock'] is None:
                    self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
                lock = kwargs.get('rw_lock', Lock())
                lock.acquire()
                if np.isfinite(score):
                    _ = self.topk_model_saver.add(config, score, regressor_id)
                lock.release()

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.logger.info('Evaluator: %s' % (str(e)))
                score = -np.inf

        elif 'partial' in self.resampling_strategy:
            try:
                # Prepare data node.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    if self.resampling_params is None or 'test_size' not in self.resampling_params:
                        test_size = 0.33
                    else:
                        test_size = self.resampling_params['test_size']

                    from sklearn.model_selection import ShuffleSplit
                    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                    for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                        _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                        _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                    self.train_node.data = [_x_train, _y_train]
                    self.val_node.data = [_x_val, _y_val]

                    data_node, op_list = parse_config(self.train_node, config, record=True)
                    _val_node = self.val_node.copy_()
                    _val_node = construct_node(_val_node, op_list)

                _x_train, _y_train = data_node.data

                if downsample_ratio != 1:
                    down_ss = ShuffleSplit(n_splits=1, test_size=downsample_ratio,
                                           random_state=self.seed)
                    for _, _val_index in down_ss.split(_x_train, _y_train):
                        _act_x_train, _act_y_train = _x_train[_val_index], _y_train[_val_index]
                else:
                    _act_x_train, _act_y_train = _x_train, _y_train
                    _val_index = list(range(len(_x_train)))

                _x_val, _y_val = _val_node.data

                config_dict = config.get_dictionary().copy()
                # Regressor gadgets
                regressor_id, clf = get_estimator(config_dict, self.estimator_id)

                score = validation(clf, self.scorer, _act_x_train, _act_y_train, _x_val, _y_val,
                                   random_state=self.seed)

                # TODO: Only save models with maximum resources
                if 'rw_lock' not in kwargs or kwargs['rw_lock'] is None:
                    self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
                lock = kwargs.get('rw_lock', Lock())
                lock.acquire()
                if np.isfinite(score) and downsample_ratio == 1:
                    save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(config, score,
                                                                                                       regressor_id)
                    if save_flag is True:
                        with open(model_path, 'wb') as f:
                            pkl.dump([op_list, clf], f)
                        self.logger.info("Model saved to %s" % model_path)

                    try:
                        if delete_flag and os.path.exists(model_path_deleted):
                            os.remove(model_path_deleted)
                            self.logger.info("Model deleted from %s" % model_path)
                    except:
                        pass
                lock.release()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.logger.info('Evaluator: %s' % (str(e)))
                score = -np.inf

        else:
            raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)

        try:
            self.logger.info('Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                             (regressor_id,
                              self.scorer._sign * score,
                              time.time() - start_time, _x_train.shape))
        except:
            pass

        # Turn it into a minimization problem.
        return_dict['score'] = -score
        return -score
