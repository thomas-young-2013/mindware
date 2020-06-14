import os
import hashlib
import numpy as np
import pickle as pk
from collections import OrderedDict
from .meta_generator import get_feature_vector, prepare_meta_dataset
from solnml.components.utils.constants import CLS_TASKS, REG_TASKS
from solnml.utils.logging_utils import get_logger

_buildin_algorithms = ['lightgbm', 'random_forest', 'libsvm_svc', 'extra_trees', 'liblinear_svc',
                       'k_nearest_neighbors', 'logistic_regression', 'gradient_boosting', 'adaboost']


class BaseAdvisor(object):
    def __init__(self, n_algorithm=3,
                 task_type=None,
                 metric='acc',
                 rep=3,
                 total_resource=20,
                 meta_algorithm='lightgbm',
                 exclude_datasets=None,
                 meta_dir=None):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.n_algorithm = n_algorithm
        self.n_algo_candidates = len(_buildin_algorithms)
        self.task_type = task_type
        self.meta_algo = meta_algorithm
        if task_type in CLS_TASKS:
            if metric not in ['acc', 'bal_acc']:
                self.logger.info('Meta information about metric-%s does not exist, use accuracy instead.' % str(metric))
                metric = 'acc'
        elif task_type in REG_TASKS:
            raise NotImplementedError()
        else:
            raise ValueError('Invalid metric: %s.' % metric)

        self.metric = metric
        self.rep = rep
        self.total_resource = total_resource
        self.meta_learner = None
        buildin_loc = os.path.dirname(__file__) + '/../meta_resource/'
        self.meta_dir = meta_dir if meta_dir is not None else buildin_loc
        self.exclude_datasets = exclude_datasets
        if self.exclude_datasets is None:
            self.hash_id = 'none'
        else:
            exclude_str = ','.join(sorted(self.exclude_datasets))
            md5 = hashlib.md5()
            md5.update(exclude_str.encode('utf-8'))
            self.hash_id = md5.hexdigest()
        meta_datasets = set()
        meta_runs_dir = self.meta_dir + 'meta_runs/%s/' % self.metric
        for _record in os.listdir(meta_runs_dir):
            if _record.endswith('.pkl') and _record.find('-') != -1:
                meta_name = '-'.join(_record.split('-')[:-4])
                if self.exclude_datasets is not None and meta_name in self.exclude_datasets:
                    continue
                meta_datasets.add(meta_name)
        self._buildin_datasets = list(meta_datasets)
        if not self.meta_dir.endswith('/'):
            self.meta_dir += '/'

    def load_train_data(self):
        meta_dataset_filename = self.meta_dir + 'ranker_%s_dataset_%s_%s.pkl' % (
            self.meta_algo, self.metric, self.hash_id)
        if os.path.exists(meta_dataset_filename):
            self.logger.info('Meta dataset file exists:', meta_dataset_filename)
            with open(meta_dataset_filename, 'rb') as f:
                meta_X, meta_y, meta_infos = pk.load(f)
        else:
            X, Y, include_datasets = prepare_meta_dataset(self.meta_dir, self.metric,
                                                          self.total_resource, self.rep,
                                                          self._buildin_datasets, _buildin_algorithms,
                                                          task_type=self.task_type)
            meta_X, meta_y = list(), list()
            self.logger.info('Meta information comes from %d datasets.' % len(meta_y))
            meta_infos = list()

            for idx in range(len(X)):
                meta_feature, run_results, _dataset = X[idx], Y[idx], include_datasets[idx]
                _instance_num = 0
                n_algo = len(run_results)
                _X, _y = list(), list()

                for i in range(n_algo):
                    vector_i = np.zeros(n_algo)
                    vector_i[i] = 1
                    meta_x = meta_feature.copy()
                    meta_x.extend(vector_i.copy())
                    meta_label = run_results[i]
                    _X.append(meta_x)
                    _y.append(meta_label)
                    _instance_num += 1

                self.logger.info('Meta instances: %s - %d' % (_dataset, _instance_num))
                meta_X.append(_X)
                meta_y.append(_y)
                meta_infos.append((_dataset, _instance_num))

            meta_X, meta_y = np.array(meta_X), np.array(meta_y)
            with open(meta_dataset_filename, 'wb') as f:
                pk.dump([meta_X, meta_y, meta_infos], f)
        return meta_X, meta_y

    def load_test_data(self, meta_feature):
        n_algo = self.n_algo_candidates
        _X = list()
        for i in range(n_algo):
            vector_i = np.zeros(n_algo)
            vector_i[i] = 1
            meta_x = meta_feature.copy()
            meta_x.extend(vector_i)
            _X.append(meta_x)
        return np.asarray(_X)

    def fetch_algorithm_set(self, dataset, dataset_id=None):
        input_vector = get_feature_vector(dataset, dataset_id, task_type=self.task_type)
        preds = self.predict(input_vector)
        idxs = np.argsort(-preds)
        return [_buildin_algorithms[idx] for idx in idxs]

    def fetch_run_results(self, dataset):
        X, Y, include_datasets = prepare_meta_dataset(self.meta_dir, self.metric,
                                                      self.total_resource, self.rep,
                                                      [dataset], _buildin_algorithms,
                                                      task_type=self.task_type)
        idxs = np.argsort(-np.array(Y[0]))
        sorted_algos = [_buildin_algorithms[idx] for idx in idxs]
        sorted_scores = [Y[0][idx] for idx in idxs]
        return OrderedDict(zip(sorted_algos, sorted_scores))

    def fit(self):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()
