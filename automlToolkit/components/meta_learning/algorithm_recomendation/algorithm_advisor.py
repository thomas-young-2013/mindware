import os
import numpy as np
import pickle as pk
import lightgbm as lgb
from collections import OrderedDict
from .meta_generator import get_feature_vector, prepare_meta_dataset

_buildin_algorithms = ['lightgbm', 'random_forest', 'libsvm_svc', 'extra_trees', 'liblinear_svc',
                       'k_nearest_neighbors', 'logistic_regression', 'gradient_boosting', 'adaboost']


class AlgorithmAdvisor(object):
    def __init__(self, n_algorithm=3,
                 task_type=None,
                 metric='acc',
                 rep=3,
                 total_resource=20,
                 meta_algorithm='lightgbm',
                 meta_dir=None):
        self.n_algorithm = n_algorithm
        self.task_type = task_type
        self.meta_algo = meta_algorithm
        self.metric = metric
        self.rep = rep
        self.total_resource = total_resource
        self.meta_dir = meta_dir if meta_dir is not None else './data/meta_res_cp'
        meta_datasets = set()
        for _record in os.listdir(self.meta_dir):
            if _record.endswith('.pkl') and _record.find('-') != -1:
                meta_name = '-'.join(_record.split('-')[:-4])
                meta_datasets.add(meta_name)
        self._buildin_datasets = list(meta_datasets)
        if not self.meta_dir.endswith('/'):
            self.meta_dir += '/'
        if not os.path.exists(self.meta_dir):
            os.makedirs(self.meta_dir)

    def fetch_algorithm_set(self, dataset, data_dir='./'):
        input_vector = get_feature_vector(dataset, data_dir, task_type=self.task_type)
        pred_algos, _ = self.predict_meta_learner(input_vector)
        return pred_algos[:self.n_algorithm]

    def fetch_run_results(self, dataset):
        X, Y, include_datasets = prepare_meta_dataset(self.meta_dir, self.metric,
                                                      self.total_resource, self.rep,
                                                      [dataset], _buildin_algorithms,
                                                      task_type=self.task_type)
        idxs = np.argsort(-np.array(Y[0]))
        sorted_algos = [_buildin_algorithms[idx] for idx in idxs]
        sorted_scores = [Y[0][idx] for idx in idxs]
        return OrderedDict(zip(sorted_algos, sorted_scores))

    def fit_meta_learner(self):
        meta_dataset_filename = self.meta_dir + '/ranker_dataset_%s.pkl' % self.meta_algo
        if os.path.exists(meta_dataset_filename):
            with open(meta_dataset_filename, 'rb') as f:
                meta_X, meta_y, meta_infos = pk.load(f)
        else:
            X, Y, include_datasets = prepare_meta_dataset(self.meta_dir, self.metric,
                                                          self.total_resource, self.rep,
                                                          self._buildin_datasets, _buildin_algorithms,
                                                          task_type=self.task_type)
            meta_X, meta_y = list(), list()

            print('Meta information comes from %d datasets.' % len(meta_y))
            meta_infos = list()
            for idx in range(len(X)):
                meta_feature, run_results, _dataset = X[idx], Y[idx], include_datasets[idx]
                print(dict(zip(_buildin_algorithms, run_results)))
                _instance_num = 0
                n_algo = len(run_results)
                for i in range(n_algo):
                    for j in range(i + 1, n_algo):
                        if run_results[i] == -1 or run_results[j] == -1:
                            continue

                        vector_i, vector_j = np.zeros(n_algo), np.zeros(n_algo)
                        vector_i[i] = 1
                        vector_j[j] = 1

                        meta_x1 = meta_feature.copy()
                        meta_x1.extend(vector_i.copy())
                        meta_x1.extend(vector_j.copy())

                        meta_x2 = meta_feature.copy()
                        meta_x2.extend(vector_j.copy())
                        meta_x2.extend(vector_i.copy())

                        meta_label1 = 1 if run_results[i] > run_results[j] else 0
                        meta_label2 = 1 - meta_label1
                        meta_X.append(meta_x1)
                        meta_y.append(meta_label1)
                        meta_X.append(meta_x2)
                        meta_y.append(meta_label2)
                        _instance_num += 1
                print('meta instances: ', _dataset, _instance_num)
                meta_infos.append((_dataset, _instance_num))

            meta_X, meta_y = np.array(meta_X), np.array(meta_y)
            with open(meta_dataset_filename, 'wb') as f:
                pk.dump([meta_X, meta_y, meta_infos], f)

        print('Starting training...')
        # train
        print(meta_X.shape, meta_y.shape)

        meta_learner_config_filename = self.meta_dir + '/meta_learner_%s_config.pkl' % self.meta_algo
        if os.path.exists(meta_learner_config_filename):
            with open(meta_learner_config_filename, 'rb') as f:
                meta_learner_config = pk.load(f)
                print('load meta-learner config from file.')
                print(meta_learner_config)
        else:
            meta_learner_config = dict()

        gbm = lgb.LGBMClassifier(**meta_learner_config)
        gbm.fit(meta_X, meta_y)

        print('Dumping model to PICKLE...')

        with open(self.meta_dir + '/ranker_model_%s.pkl' % self.meta_algo, 'wb') as f:
            pk.dump(gbm, f)
        return meta_infos

    def predict_meta_learner(self, meta_feature):
        with open(self.meta_dir + '/ranker_model_%s.pkl' % self.meta_algo, 'rb') as f:
            gbm = pk.load(f)

            n_algo = len(_buildin_algorithms)
            _X = list()
            for i in range(n_algo):
                for j in range(i + 1, n_algo):
                    vector_i, vector_j = np.zeros(n_algo), np.zeros(n_algo)
                    vector_i[i] = 1
                    vector_j[j] = 1

                    meta_x = meta_feature.copy()
                    meta_x.extend(vector_i)
                    meta_x.extend(vector_j)
                    _X.append(meta_x)

            preds = gbm.predict(_X)
            # print(preds)

            instance_idx = 0
            scores = np.zeros(n_algo)
            for i in range(n_algo):
                for j in range(i + 1, n_algo):
                    if preds[instance_idx] == 1:
                        scores[i] += 1
                    else:
                        scores[j] += 1
                    instance_idx += 1
            scores = np.array(scores) / np.sum(scores)
            idxs = np.argsort(-scores)
            sorted_algos = [_buildin_algorithms[idx] for idx in idxs]
            sorted_scores = [scores[idx] for idx in idxs]
            return sorted_algos, sorted_scores
