import os
import numpy as np
import pickle as pk
import lightgbm as lgb
from solnml.utils.logging_utils import get_logger
from solnml.components.meta_learning.algorithm_recomendation.base_advisor import BaseAdvisor


class GBMAdvisor(BaseAdvisor):
    def __init__(self, n_algorithm=3,
                 task_type=None,
                 metric='acc',
                 exclude_datasets=None):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        super().__init__(n_algorithm, task_type, metric=metric,
                         meta_algorithm='lightgbm', exclude_datasets=exclude_datasets)
        self.model = None

    @staticmethod
    def create_pairwise_data(X, y):
        n_algo = y.shape[1]
        X1, labels = list(), list()
        _instance_num = 0

        for _X, _y in zip(X, y):
            if np.isnan(_X).any():
                continue
            meta_vec = _X
            for i in range(n_algo):
                for j in range(i+1, n_algo):
                    if (_y[i] == -1) or (_y[j] == -1):
                        continue

                    vector_i, vector_j = np.zeros(n_algo), np.zeros(n_algo)
                    vector_i[i] = 1
                    vector_j[j] = 1

                    meta_x1 = list(meta_vec.copy())
                    meta_x1.extend(vector_i.copy())
                    meta_x1.extend(vector_j.copy())

                    meta_x2 = meta_vec.copy()
                    meta_x2.extend(vector_j.copy())
                    meta_x2.extend(vector_i.copy())

                    meta_label1 = 1 if _y[i] > _y[j] else 0
                    meta_label2 = 1 - meta_label1
                    X1.append(meta_x1)
                    labels.append(meta_label1)
                    X1.append(meta_x2)
                    labels.append(meta_label2)
                    _instance_num += 1
        return np.asarray(X1), np.asarray(labels)

    def fit(self, **meta_learner_config):
        _X, _y, _ = self.metadata_manager.load_meta_data()
        print(_X.shape, _y.shape)
        X, y = self.create_pairwise_data(_X, _y)

        # meta_learner_config_filename = self.meta_dir + 'meta_learner_%s_%s_%s_config.pkl' % (
        #     self.meta_algo, self.metric, 'none')
        # if os.path.exists(meta_learner_config_filename):
        #     with open(meta_learner_config_filename, 'rb') as f:
        #         meta_learner_config = pk.load(f)
        # print(meta_learner_config)
        self.model = lgb.LGBMClassifier(**meta_learner_config)
        self.model.fit(X, y)

    def predict(self, meta_feature):
        n_algo = self.n_algo_candidates
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

        preds = self.model.predict(_X)

        instance_idx = 0
        scores = np.zeros(n_algo)
        for i in range(n_algo):
            for j in range(i + 1, n_algo):
                if preds[instance_idx] == 1:
                    scores[i] += 1
                else:
                    scores[j] += 1
                instance_idx += 1
        return np.array(scores) / np.sum(scores)
