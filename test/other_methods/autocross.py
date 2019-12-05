import numpy as np
import math

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
import logging
from time import time

# some the constants in autocross
GAMMA = 1.5  # gamma is the decay rate to control the initial number of configurations in hyperband
SAMPLES_PER_PARTITION = 243  # samples in a partition which is '1' resource
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TreeNode:

    def __init__(self, feature_key):
        self.feature_key = feature_key
        self.feature_set = [int(feature) for feature in feature_key.split(',')]
        self.performance = 0.0

    def _get_performance(self, metricstr, train_x, train_y, valid_x, valid_y, model):
        metric = get_metric(metricstr)
        model.fit(train_x, train_y)
        if metricstr == 'auc':
            pred = model.predict_proba(valid_x)[:, 1]
        else:
            pred = model.predict(valid_x)
        return metric(pred, valid_y)

    def set_performance(self, metricstr, model, train_data, train_label, valid_data, valid_label):
        self.performance = self._get_performance(metricstr, train_data, train_label,
                                                 valid_data, valid_label, model=model)

    def __lt__(self, other):
        return self.performance < other.performance

    def __le__(self, other):
        return self.performance <= other.performance

    def __gt__(self, other):
        return self.performance > other.performance

    def __ge__(self, other):
        return self.performance >= other.performance


"""
AutoCross is a feature generation algorithm based on the KDD 2019 paper.
"""


class AutoCross:

    def __init__(self, max_iter, metrics, model=LogisticRegression(multi_class='auto', solver='liblinear')):
        '''

        :param max_iter: Maximum search iteration
        :param metrics:
        :param model:
        '''
        self.max_iter = max_iter
        self.model = model
        self.metricstr = metrics
        self.metric = get_metric(metrics)
        self.feature_sets = None
        self.feature_cols = dict()
        self.train_data = None
        self.valid_data = None
        self.numerical_features = None
        self.init_length = None
        self.kbins_discretizer_5 = KBinsDiscretizer(n_bins=5, strategy='uniform')
        self.kbins_discretizer_50 = KBinsDiscretizer(n_bins=50, strategy='uniform')
        self.onehot_encoder = OneHotEncoder()
        self.numerical_index = None
        self.categorical_index = None
        self.onehot_index = None

    def _get_cross_feature_val(self, feature_set, x):
        assert len(feature_set) >= 1
        feature_val = np.ones(len(x), dtype=np.float32)
        for feature_id in feature_set:
            feature_val *= x[:, feature_id]
        return feature_val.reshape((-1, 1))

    def _get_performance(self, train_x, train_y, valid_x, valid_y, model):
        metric = get_metric(self.metricstr)
        model.fit(train_x, train_y)
        if self.metricstr == 'auc':
            pred = model.predict_proba(valid_x)[:, 1]
        else:
            pred = model.predict(valid_x)

        return metric(pred, valid_y)

    def _get_stratify_sample(self, x, y, n_samples):
        assert len(x) == len(y)
        _, x_sample, _, y_sample = train_test_split(x, y, test_size=n_samples / len(x), stratify=y)
        return x_sample, y_sample

    # def _successive_halving(self, x_train, x_valid, y_train, y_valid):
    #     feature_num = x_train.shape[1]
    #     self.feature_sets = [[feature_id] for feature_id in range(feature_num)]
    #     for feature_id in range(feature_num):
    #         self.feature_cols[str(feature_id)] = dict()
    #         self.feature_cols[str(feature_id)]["train"] = x_train[:, feature_id]
    #         self.feature_cols[str(feature_id)]["valid"] = x_valid[:, feature_id]
    #
    #     for iteration in range(self.max_iter):
    #         t0 = time()
    #         nodes = list()
    #         for i in range(len(self.feature_sets)):
    #             for j in range(i + 1, len(self.feature_sets)):
    #                 set1 = self.feature_sets[i]
    #                 set2 = self.feature_sets[j]
    #                 cross_set = set1 + set2
    #                 feature_key = ','.join(str(feature) for feature in cross_set)
    #
    #                 if self.feature_cols.get(feature_key) is None:
    #                     set1_key = ','.join(str(feature) for feature in set1)
    #                     set2_key = ','.join(str(feature) for feature in set2)
    #
    #                     # print(self.feature_cols[set1_key]["train"].shape)
    #                     # print(self.feature_cols[set2_key]["train"].shape)
    #                     self.feature_cols[feature_key] = dict()
    #                     self.feature_cols[feature_key]["train"] = \
    #                         self.feature_cols[set1_key]["train"] * self.feature_cols[set2_key]["train"]
    #                     self.feature_cols[feature_key]["valid"] = \
    #                         self.feature_cols[set1_key]["valid"] * self.feature_cols[set2_key]["valid"]
    #
    #                 node = TreeNode(feature_key=feature_key)
    #                 nodes.append(node)
    #
    #         # start the successive halving process
    #         n = len(nodes)  # n is the number of the configurations / bandits
    #         B = len(x_train)
    #         r = int(B / n)
    #         if r < 50:
    #             r = 100
    #         while n > 1 and r <= B:
    #             print("n =", n, "r =", r)
    #             n_samples = np.random.choice(B, r, replace=False)
    #             for node in nodes:
    #                 if self.train_data is None:
    #                     train_data = self.feature_cols[node.feature_key]["train"].reshape((-1, 1))[n_samples]
    #                 else:
    #                     train_data = \
    #                         np.hstack((self.train_data,
    #                                    self.feature_cols[node.feature_key]["train"].reshape((-1, 1))))[n_samples]
    #
    #                 if self.valid_data is None:
    #                     valid_data = self.feature_cols[node.feature_key]["valid"].reshape((-1, 1))
    #                 else:
    #                     valid_data = \
    #                         np.hstack((self.valid_data,
    #                                    self.feature_cols[node.feature_key]["valid"].reshape((-1, 1))))
    #
    #                 node.set_performance(model=self.model, train_data=train_data, train_label=y_train[n_samples],
    #                                      valid_data=valid_data, valid_label=y_valid, metricstr=self.metricstr)
    #             nodes.sort(reverse=True)
    #             if int(n / 2) >= 1:
    #                 nodes = nodes[:int(n / 2)]
    #             n /= 2
    #             r *= 2
    #
    #         best_node = nodes[0]
    #         best_feature_list = [int(feature) for feature in best_node.feature_key.split(',')]
    #         self.feature_sets.append(best_feature_list)
    #         if self.train_data is None:
    #             self.train_data = self.feature_cols[best_node.feature_key]["train"].reshape(-1, 1)
    #         else:
    #             self.train_data = \
    #                 np.hstack((self.train_data, self.feature_cols[best_node.feature_key]["train"].reshape(-1, 1)))
    #
    #         if self.valid_data is None:
    #             self.valid_data = self.feature_cols[best_node.feature_key]["valid"].reshape(-1, 1)
    #         else:
    #             self.valid_data = \
    #                 np.hstack((self.valid_data, self.feature_cols[best_node.feature_key]["valid"].reshape(-1, 1)))
    #
    #         self.model.fit(np.hstack((x_train, self.train_data)), y_train)
    #         y_pred = self.model.predict(np.hstack((x_valid, self.valid_data)))
    #         perf = self.metric(y_valid, y_pred)
    #         # if iteration % 5 == 0:
    #         print("iteration:", iteration, " performance:", best_node.performance, "all data perf:", perf, " time:",
    #               time() - t0)

    def _hyperband(self, x_train, y_train, categorical_index=[], numerical_index=[], eta=3,
                   early_stop_iter=5):
        x_train_copy = x_train[:]
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.66, stratify=y_train)

        feature_num = x_train.shape[1]
        id_list = list(range(feature_num))
        one_hot_index = list(set(id_list) - set(categorical_index) - set(numerical_index))
        self.onehot_index = one_hot_index

        ori_train_onehot = np.ndarray((len(x_train), 0))
        ori_valid_onehot = np.ndarray((len(x_valid), 0))
        one_hot_train_5 = np.ndarray((len(x_train), 0))
        one_hot_valid_5 = np.ndarray((len(x_valid), 0))
        one_hot_train_50 = np.ndarray((len(x_train), 0))
        one_hot_valid_50 = np.ndarray((len(x_valid), 0))
        one_hot_train_cat = np.ndarray((len(x_train), 0))
        one_hot_valid_cat = np.ndarray((len(x_valid), 0))

        if one_hot_index:
            ori_train_onehot = x_train[:, one_hot_index]
            ori_valid_onehot = x_valid[:, one_hot_index]

        if categorical_index:
            self.onehot_encoder.fit(x_train_copy[:, categorical_index])
            one_hot_train_cat = self.onehot_encoder.transform(x_train[:, categorical_index]).toarray()
            one_hot_valid_cat = self.onehot_encoder.transform(x_valid[:, categorical_index]).toarray()

        if numerical_index:
            one_hot_train_5 = self.kbins_discretizer_5.fit_transform(x_train[:, numerical_index]).toarray()
            one_hot_valid_5 = self.kbins_discretizer_5.transform(x_valid[:, numerical_index]).toarray()

            one_hot_train_50 = self.kbins_discretizer_50.fit_transform(x_train[:, numerical_index]).toarray()
            one_hot_valid_50 = self.kbins_discretizer_50.transform(x_valid[:, numerical_index]).toarray()

        self.train_data = np.hstack((ori_train_onehot, one_hot_train_5, one_hot_train_50, one_hot_train_cat))
        self.valid_data = np.hstack((ori_valid_onehot, one_hot_valid_5, one_hot_valid_50, one_hot_valid_cat))

        self.init_length = self.train_data.shape[1]
        logger.info("Feature num: " + str(self.init_length))
        self.feature_sets = [{feature_id} for feature_id in range(self.init_length)]
        global_best_perf = -1
        early_stopping_cnt = 0
        for iteration in range(self.max_iter):
            # t0 = time()
            # start the hyperband process
            R = len(self.train_data)
            s_max = int(math.log(R, eta))
            # B = (s_max + 1) * R
            best_node = None
            for s in range(s_max, -1, -1):
                nodes = []
                subset_records = set()  # record the subset
                # n = math.ceil((B / R) * (pow(eta, s) / (s + 1)))
                configurations_num = len(self.feature_sets) * (len(self.feature_sets) - 1) // 2
                n = int(configurations_num / GAMMA / pow(eta, s_max - s))
                r = int(R * pow(eta, -s)) * int(R / SAMPLES_PER_PARTITION)
                r = max(r, 50)
                if n <= 1 or r > R:
                    continue
                # We can't use field-wise LR, so we use random choice instead to reduce the high computational cost
                logger.info(n)
                while len(nodes) < n:
                    p1, p2 = np.random.choice(len(self.feature_sets), 2, replace=False)
                    set1 = self.feature_sets[p1]
                    set2 = self.feature_sets[p2]
                    cross_set = set1.union(set2)
                    feature_key = ','.join(str(feature) for feature in cross_set)
                    for ele in set1:
                        if ele in set2:
                            print(set1, set2)
                            print(feature_key)
                    if feature_key in subset_records:
                        continue
                    subset_records.add(feature_key)

                    node = TreeNode(feature_key=feature_key)
                    nodes.append(node)

                # start the inner loop
                while n > 1 and r <= R:
                    for i, node in enumerate(nodes):
                        train_data = np.hstack((self.train_data,
                                                self._get_cross_feature_val(node.feature_set, self.train_data)))
                        train_data, train_label = self._get_stratify_sample(train_data, y_train, r)
                        valid_data = np.hstack((self.valid_data,
                                                self._get_cross_feature_val(node.feature_set, self.valid_data)))
                        node.set_performance(model=self.model, train_data=train_data, train_label=train_label,
                                             valid_data=valid_data, valid_label=y_valid, metricstr=self.metricstr)
                        logger.info("LR trained: %d/%d with performance %f" % (i, len(nodes), node.performance))

                    nodes.sort(reverse=True)
                    n = math.ceil(n / eta)
                    r = int(r * eta)
                    if n >= 1:
                        nodes = nodes[:n]

                if best_node is None:
                    best_node = nodes[0]
                else:
                    if nodes[0].performance > best_node.performance:
                        best_node = nodes[0]

                # print("outer loop:", iteration, "inner loop:", s, "performance:",
                #       best_node.performance, " time:",
                #       time() - t0, flush=True)

            if best_node.performance > global_best_perf:
                global_best_perf = best_node.performance
                early_stopping_cnt = 0

                # update train_data and valid_data
                best_feature_list = {int(feature) for feature in best_node.feature_key.split(',')}
                self.feature_sets.append(best_feature_list)
                new_train_feature = self._get_cross_feature_val(best_node.feature_set, self.train_data)
                new_valid_feature = self._get_cross_feature_val(best_node.feature_set, self.valid_data)
                self.train_data = np.hstack((self.train_data, new_train_feature))
                self.valid_data = np.hstack((self.valid_data, new_valid_feature))
            else:
                early_stopping_cnt += 1
                if early_stopping_cnt == early_stop_iter:
                    logger.info("no improvement for", early_stopping_cnt, "iterations..........")
                    break

            logger.info("generate feature: " + str(iteration) + " global performance: " + str(global_best_perf))

            np.savez("features_" + str(iteration), train=self.train_data, valid=self.valid_data)

    def fit(self, x_train, y_train, categorical_index=[], numerical_index=[]):
        '''

        :param x_train: features
        :param y_train: labels
        :param categorical_index: indices of categorical features
        :param numerical_index: indices of numerical features
        :return:
        '''
        self.categorical_index = categorical_index
        self.numerical_index = numerical_index
        self._hyperband(x_train=x_train, y_train=y_train,
                        categorical_index=categorical_index,
                        numerical_index=numerical_index)

    def transform(self, x):
        '''

        :param x: features to be transformed
        :return: Only the generated features (without original features)
        '''
        ori_onehot = np.ndarray((len(x), 0))
        one_hot_5 = np.ndarray((len(x), 0))
        one_hot_50 = np.ndarray((len(x), 0))
        one_hot_cat = np.ndarray((len(x), 0))
        if self.onehot_index:
            ori_onehot = x[:, self.onehot_index]
        if self.numerical_index:
            one_hot_5 = self.kbins_discretizer_5.transform(x[:, self.numerical_index]).toarray()
            one_hot_50 = self.kbins_discretizer_50.transform(x[:, self.numerical_index]).toarray()
        if self.categorical_index:
            one_hot_cat = self.onehot_encoder.transform(x[:, self.categorical_index]).toarray()

        pri_data = np.hstack((ori_onehot, one_hot_5, one_hot_50, one_hot_cat))
        generated_features = None
        # Collect only new features
        for feature_set in self.feature_sets[self.init_length:]:
            if generated_features is None:
                generated_features = self._get_cross_feature_val(feature_set, pri_data)
            else:
                generated_features = np.hstack((generated_features, self._get_cross_feature_val(feature_set, pri_data)))
        return generated_features


def get_metric(metricstr):
    # Metrics for classification
    if metricstr in ["accuracy", "acc"]:
        from sklearn.metrics import accuracy_score
        return accuracy_score
    elif metricstr == 'f1':
        from sklearn.metrics import f1_score
        return f1_score
    elif metricstr == 'precision':
        from sklearn.metrics import precision_score
        return precision_score
    elif metricstr == 'recall':
        from sklearn.metrics import recall_score
        return recall_score
    elif metricstr == "auc":
        from sklearn.metrics import roc_auc_score
        return roc_auc_score

    # Metrics for regression
    elif metricstr in ["mean_squared_error", "mse"]:
        from sklearn.metrics import mean_squared_error
        return mean_squared_error
    elif metricstr in ['mean_squared_log_error', "msle"]:
        from sklearn.metrics import mean_squared_log_error
        return mean_squared_log_error
    elif metricstr == "evs":
        from sklearn.metrics import explained_variance_score
        return explained_variance_score
    elif metricstr == "r2":
        from sklearn.metrics import r2_score
        return r2_score
    elif metricstr in ["mean_absolute_error", "mae"]:
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error
    elif callable(metricstr):
        return metricstr
    else:
        raise ValueError("Given", metricstr, ". Expected valid metric string like 'acc' or callable metric function!")
