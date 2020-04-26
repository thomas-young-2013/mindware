import os
import numpy as np
import pickle as pk
import lightgbm as lgb
from .meta_generator import get_feature_vector, prepare_meta_dataset

_buildin_datasets = ['diabetes', 'fri_c1', 'ionosphere']
_buildin_algorithms = ['lightgbm', 'random_forest', 'libsvm_svc', 'extra_trees', 'liblinear_svc',
                       'k_nearest_neighbors', 'logistic_regression', 'gradient_boosting', 'adaboost']


class AlgorithmAdvisor(object):
    def __init__(self, n_algorithm=3,
                 metric='acc',
                 rep=3,
                 total_resource=20,
                 meta_algorithm='lightgbm',
                 meta_dir=None):
        self.n_algorithm = n_algorithm
        self.meta_algo = meta_algorithm
        self.metric = metric
        self.rep = rep
        self.total_resource = total_resource
        self.meta_dir = meta_dir if meta_dir is not None else './data/meta_res_cp'
        if not self.meta_dir.endswith('/'):
            self.meta_dir += '/'
        if not os.path.exists(self.meta_dir):
            os.makedirs(self.meta_dir)

    def fetch_algorithm_set(self, dataset, data_dir='./'):
        input_vector = get_feature_vector(dataset, data_dir)
        _, pred_scores = self.predict_meta_learner(input_vector)
        algo_idx = np.argsort(-pred_scores)
        return [_buildin_algorithms[idx] for idx in algo_idx[:self.n_algorithm]]

    def fit_meta_learner(self):
        X, Y = prepare_meta_dataset(self.meta_dir, self.metric,
                                    self.total_resource, self.rep,
                                    _buildin_datasets, _buildin_algorithms)
        meta_X, meta_y = list(), list()

        for meta_feature, run_results in zip(X, Y):
            n_algo = len(run_results)
            for i in range(n_algo):
                for j in range(i+1, n_algo):
                    vector_i, vector_j = np.zeros(n_algo), np.zeros(n_algo)
                    vector_i[i] = 1
                    vector_j[j] = 1

                    meta_x = meta_feature.copy()
                    meta_x.extend(vector_i)
                    meta_x.extend(vector_j)

                    meta_label = 1 if run_results[i] > run_results[j] else 0
                    meta_X.append(meta_x)
                    meta_y.append(meta_label)

        print('Starting training...')
        # train
        print(np.array(meta_X).shape)
        print(np.array(meta_y).shape)
        gbm = lgb.LGBMRegressor(num_leaves=31,
                                learning_rate=0.05,
                                n_estimators=100)
        gbm.fit(meta_X, meta_y)

        print('Dumping model to PICKLE...')

        with open(self.meta_dir+'/ranker_model_%s.pkl' % self.meta_algo, 'wb') as f:
            pk.dump(gbm, f)
        return self

    def predict_meta_learner(self, meta_feature):
        with open(self.meta_dir+'/ranker_model_%s.pkl' % self.meta_algo, 'rb') as f:
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

            instance_idx = 0
            scores = np.zeros(n_algo)
            for i in range(n_algo):
                for j in range(i + 1, n_algo):
                    if preds[instance_idx] == 1:
                        scores[i] += 1
                    else:
                        scores[j] += 1
                    instance_idx += 1
            return _buildin_algorithms, np.array(scores)/np.sum(scores)
