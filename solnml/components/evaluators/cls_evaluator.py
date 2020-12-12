import os
import sys
import time
import warnings
import numpy as np
import pickle as pkl
from multiprocessing import Lock
from sklearn.metrics.scorer import balanced_accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

from solnml.utils.logging_utils import get_logger
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.evaluators.evaluate_func import validation
from solnml.components.fe_optimizers.parse import parse_config, construct_node
from solnml.components.evaluators.base_evaluator import BanditTopKModelSaver
from solnml.components.utils.class_loader import get_combined_candidtates


def get_estimator(config, estimator_id):
    from solnml.components.models.classification import _classifiers, _addons
    _candidates = get_combined_candidtates(_classifiers, _addons)
    classifier_type = estimator_id
    config_ = config.copy()
    config_['random_state'] = 1
    estimator = _candidates[classifier_type](**config_)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', 1)
    return classifier_type, estimator


class ClassificationEvaluator(_BaseEvaluator):
    def __init__(self, clf_config, fe_config, estimator_id, if_imbal=False, scorer=None, data_node=None, name=None,
                 resampling_strategy='cv', resampling_params=None, seed=1,
                 timestamp=None, output_dir=None):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params
        self.hpo_config = clf_config

        # TODO: Optimize: Fit the same transformers only once
        self.fe_config = fe_config
        self.estimator_id = estimator_id
        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.if_imbal = if_imbal
        self.data_node = data_node
        self.name = name
        self.seed = seed
        self.onehot_encoder = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        self.output_dir = output_dir
        self.timestamp = timestamp

        self.train_node = data_node.copy_()
        self.val_node = data_node.copy_()

        self.continue_training = False

        self.topk_model_saver = BanditTopKModelSaver(k=60, model_dir=self.output_dir, identifier=timestamp)

    def get_fit_params(self, y, estimator):
        from solnml.components.utils.balancing import get_weights
        _init_params, _fit_params = get_weights(
            y, estimator, None, {}, {})
        return _init_params, _fit_params

    def __call__(self, config, **kwargs):
        start_time = time.time()
        return_dict = dict()

        if self.name is None:
            raise ValueError('This evaluator has no name/type!')
        assert self.name in ['hpo', 'fe']

        if self.name == 'hpo':
            hpo_config = config if config is not None else self.hpo_config
            fe_config = kwargs.get('ano_config', self.fe_config)
        else:
            fe_config = config if config is not None else self.fe_config
            hpo_config = kwargs.get('ano_config', self.hpo_config)

        # Prepare configuration.
        self.seed = 1
        downsample_ratio = kwargs.get('resource_ratio', 1.0)

        if 'holdout' in self.resampling_strategy:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    if self.resampling_params is None or 'test_size' not in self.resampling_params:
                        test_size = 0.33
                    else:
                        test_size = self.resampling_params['test_size']

                    from sklearn.model_selection import StratifiedShuffleSplit
                    ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                    for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                        _X_train, _X_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                        _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                    self.train_node.data = [_X_train, _y_train]
                    self.val_node.data = [_X_val, _y_val]

                    data_node, op_list = parse_config(self.train_node, fe_config, record=True, if_imbal=self.if_imbal)
                    _val_node = self.val_node.copy_()
                    _val_node = construct_node(_val_node, op_list)

                _X_train, _y_train = data_node.data
                _X_val, _y_val = _val_node.data

                config_dict = hpo_config.get_dictionary().copy()
                # Prepare training and initial params for classifier.
                init_params, fit_params = {}, {}
                if data_node.enable_balance == 1:
                    init_params, fit_params = self.get_fit_params(_y_train, self.estimator_id)
                    for key, val in init_params.items():
                        config_dict[key] = val

                if data_node.data_balance == 1:
                    fit_params['data_balance'] = True

                classifier_id, clf = get_estimator(config_dict, self.estimator_id)

                if self.onehot_encoder is None:
                    self.onehot_encoder = OneHotEncoder(categories='auto')
                    y = np.reshape(_y_train, (len(_y_train), 1))
                    self.onehot_encoder.fit(y)

                score = validation(clf, self.scorer, _X_train, _y_train, _X_val, _y_val,
                                   random_state=self.seed,
                                   onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                            _ThresholdScorer) else None,
                                   fit_params=fit_params)

                if 'rw_lock' not in kwargs or kwargs['rw_lock'] is None:
                    self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
                lock = kwargs.get('rw_lock', Lock())
                lock.acquire()
                if np.isfinite(score):
                    save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(hpo_config,
                                                                                                       fe_config,
                                                                                                       score,
                                                                                                       classifier_id)
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
                self.logger.info('%s-evaluator: %s' % (self.name, str(e)))
                score = -np.inf
                traceback.print_exc(file=sys.stdout)

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

                    from sklearn.model_selection import StratifiedKFold
                    skfold = StratifiedKFold(n_splits=folds, random_state=self.seed, shuffle=False)
                    scores = list()

                    for train_index, test_index in skfold.split(self.data_node.data[0], self.data_node.data[1]):
                        _X_train, _X_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                        _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                        self.train_node.data = [_X_train, _y_train]
                        self.val_node.data = [_X_val, _y_val]

                        data_node, op_list = parse_config(self.train_node, fe_config, record=True,
                                                          if_imbal=self.if_imbal)
                        _val_node = self.val_node.copy_()
                        _val_node = construct_node(_val_node, op_list)

                        _X_train, _y_train = data_node.data
                        _X_val, _y_val = _val_node.data

                        config_dict = hpo_config.get_dictionary().copy()
                        # Prepare training and initial params for classifier.
                        init_params, fit_params = {}, {}
                        if data_node.enable_balance == 1:
                            init_params, fit_params = self.get_fit_params(_y_train, self.estimator_id)
                            for key, val in init_params.items():
                                config_dict[key] = val

                        if data_node.data_balance == 1:
                            fit_params['data_balance'] = True

                        classifier_id, clf = get_estimator(config_dict, self.estimator_id)

                        if self.onehot_encoder is None:
                            self.onehot_encoder = OneHotEncoder(categories='auto')
                            y = np.reshape(_y_train, (len(_y_train), 1))
                            self.onehot_encoder.fit(y)

                        _score = validation(clf, self.scorer, _X_train, _y_train, _X_val, _y_val,
                                            random_state=self.seed,
                                            onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                     _ThresholdScorer) else None,
                                            fit_params=fit_params)
                        scores.append(_score)
                    score = np.mean(scores)

                # TODO: Don't save models for cv
                if 'rw_lock' not in kwargs or kwargs['rw_lock'] is None:
                    self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
                lock = kwargs.get('rw_lock', Lock())
                lock.acquire()
                if np.isfinite(score):
                    _ = self.topk_model_saver.add(hpo_config, fe_config, score, classifier_id)
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

                    from sklearn.model_selection import StratifiedShuffleSplit
                    ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                    for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                        _X_train, _X_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                        _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                    self.train_node.data = [_X_train, _y_train]
                    self.val_node.data = [_X_val, _y_val]

                    data_node, op_list = parse_config(self.train_node, fe_config, record=True, if_imbal=self.if_imbal)
                    _val_node = self.val_node.copy_()
                    _val_node = construct_node(_val_node, op_list)

                _X_train, _y_train = data_node.data

                if downsample_ratio != 1:
                    down_ss = StratifiedShuffleSplit(n_splits=1, test_size=downsample_ratio,
                                                     random_state=self.seed)
                    for _, _val_index in down_ss.split(_X_train, _y_train):
                        _act_x_train, _act_y_train = _X_train[_val_index], _y_train[_val_index]
                else:
                    _act_x_train, _act_y_train = _X_train, _y_train
                    _val_index = list(range(len(_X_train)))

                _X_val, _y_val = _val_node.data

                config_dict = hpo_config.get_dictionary().copy()
                # Prepare training and initial params for classifier.
                init_params, fit_params = {}, {}
                if data_node.enable_balance == 1:
                    init_params, fit_params = self.get_fit_params(_y_train, self.estimator_id)
                    for key, val in init_params.items():
                        config_dict[key] = val
                if 'sample_weight' in fit_params:
                    fit_params['sample_weight'] = fit_params['sample_weight'][_val_index]
                if data_node.data_balance == 1:
                    fit_params['data_balance'] = True

                classifier_id, clf = get_estimator(config_dict, self.estimator_id)

                if self.onehot_encoder is None:
                    self.onehot_encoder = OneHotEncoder(categories='auto')
                    y = np.reshape(_y_train, (len(_y_train), 1))
                    self.onehot_encoder.fit(y)

                score = validation(clf, self.scorer, _act_x_train, _act_y_train, _X_val, _y_val,
                                   random_state=self.seed,
                                   onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                            _ThresholdScorer) else None,
                                   fit_params=fit_params)

                # TODO: Only save models with maximum resources
                if 'rw_lock' not in kwargs or kwargs['rw_lock'] is None:
                    self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
                lock = kwargs.get('rw_lock', Lock())
                lock.acquire()
                if np.isfinite(score) and downsample_ratio == 1:
                    save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(hpo_config,
                                                                                                       fe_config, score,
                                                                                                       classifier_id)
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
                             (classifier_id,
                              self.scorer._sign * score,
                              time.time() - start_time, _X_train.shape))
        except:
            pass

        # Turn it into a minimization problem.
        return_dict['score'] = -score
        return -score
