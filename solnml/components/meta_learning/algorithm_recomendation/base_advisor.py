import os
import hashlib
import numpy as np
import pickle as pk
from collections import OrderedDict
from solnml.utils.logging_utils import get_logger
from solnml.components.utils.constants import CLS_TASKS, REG_TASKS
from solnml.components.meta_learning.algorithm_recomendation.metadata_manager import MetaDataManager
from solnml.components.meta_learning.algorithm_recomendation.metadata_manager import get_feature_vector


_builtin_algorithms = ['lightgbm', 'random_forest', 'libsvm_svc', 'extra_trees', 'liblinear_svc',
                       'k_nearest_neighbors', 'adaboost', 'lda', 'qda']


class BaseAdvisor(object):
    def __init__(self, n_algorithm=3,
                 task_type=None,
                 metric='bal_acc',
                 rep=3,
                 total_resource=1200,
                 meta_algorithm='lightgbm',
                 exclude_datasets=None,
                 meta_dir=None):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.n_algorithm = n_algorithm
        self.n_algo_candidates = len(_builtin_algorithms)
        self.task_type = task_type
        self.meta_algo = meta_algorithm
        self.rep = rep
        self.metric = metric
        if task_type in CLS_TASKS:
            if metric not in ['acc', 'bal_acc']:
                self.logger.info('Meta information about metric-%s does not exist, use accuracy instead.' % str(metric))
                metric = 'acc'
        elif task_type in REG_TASKS:
            raise NotImplementedError()
        else:
            raise ValueError('Invalid metric: %s.' % metric)

        self.total_resource = total_resource
        self.exclude_datasets = exclude_datasets

        builtin_loc = os.path.dirname(__file__)
        builtin_loc = os.path.join(builtin_loc, '..')
        builtin_loc = os.path.join(builtin_loc, 'meta_resource')
        self.meta_dir = meta_dir if meta_dir is not None else builtin_loc

        if self.exclude_datasets is None:
            self.hash_id = 'none'
        else:
            self.exclude_datasets = list(set(exclude_datasets))
            exclude_str = ','.join(sorted(self.exclude_datasets))
            md5 = hashlib.md5()
            md5.update(exclude_str.encode('utf-8'))
            self.hash_id = md5.hexdigest()
        meta_datasets = set()
        _folder = os.path.join(self.meta_dir, 'meta_runs')
        meta_runs_dir = os.path.join(_folder, self.metric)
        for _record in os.listdir(meta_runs_dir):
            if _record.endswith('.pkl') and _record.find('-') != -1:
                meta_name = '-'.join(_record.split('-')[:-4])
                if self.exclude_datasets is not None and meta_name in self.exclude_datasets:
                    continue
                meta_datasets.add(meta_name)
        self._builtin_datasets = list(meta_datasets)

        self.metadata_manager = MetaDataManager(self.meta_dir, _builtin_algorithms, self._builtin_datasets,
                                                metric, total_resource, task_type=task_type, rep=rep)
        self.meta_learner = None

    @DeprecationWarning
    def load_train_data(self):
        file_id = 'ranker_%s_dataset_%s_%s.pkl' % (self.meta_algo, self.metric, self.hash_id)
        meta_dataset_filename = os.path.join(self.meta_dir, file_id)
        if os.path.exists(meta_dataset_filename):
            self.logger.info('Meta dataset file exists: %s' % meta_dataset_filename)
            with open(meta_dataset_filename, 'rb') as f:
                meta_X, meta_y, meta_infos = pk.load(f)
        else:
            X, Y, include_datasets = self.metadata_manager.load_meta_data()
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

    @DeprecationWarning
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
        return [_builtin_algorithms[idx] for idx in idxs]

    @DeprecationWarning
    def fetch_run_results(self, dataset):
        X, Y, include_datasets = self.metadata_manager.load_meta_data()
        idxs = np.argsort(-np.array(Y[0]))
        sorted_algos = [_builtin_algorithms[idx] for idx in idxs]
        sorted_scores = [Y[0][idx] for idx in idxs]
        return OrderedDict(zip(sorted_algos, sorted_scores))

    def fit(self):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()
