from sklearn.neural_network import MLPClassifier as MLP
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.metrics import f1_score
import pickle as pkl
import numpy as np
import os
import warnings
from collections import Counter

from operators.unary import *

warnings.filterwarnings("ignore")
exclude_dataset = ['fbis_wc']


class LFE(object):  # Learning Feature Engineering for Classiﬁcation, IJCAI 2017
    def __init__(self, lower=-10, upper=10, num_bins=200, theta=0.01, gamma=0.8):  # gamma not mentioned in the work
        '''

        :param lower: lower bound
        :param upper: upper bound
        :param num_bins: number of bins
        :param theta: threshold for deciding whether a sample is positive
        :param gamma: threshold for deciding whether to recommend the best transformation. If prediction > threshold, recommend!
        '''
        self.lower = lower
        self.upper = upper
        self.num_bins = num_bins
        self.theta = theta
        self.gamma = gamma
        self.name_prefix = "lower_" + str(lower) + "_upper_" + str(upper) + "_bins_" + str(num_bins) + "_theta_" + str(
            theta)

    def generate_samples(self, x, y, dataset_name, save_dir='lfe/data'):  # One-vs-rest
        '''
        Given a dataset, generate training samples for LFE
        :param x: features
        :param y: labels
        :param dataset_name: dataset name
        :return: QSA meta-features, <transformation, label list> dicitonary like {'log':[0,1,0],'sigmoid':[1,1,1]}
        '''
        if not os.path.exists(save_dir):
            raise ValueError("Directory %s not existed!" % save_dir)
        qsa_save_path = os.path.join(save_dir, "qsa_" + dataset_name)
        label_save_path = os.path.join(save_dir, "label_" + dataset_name)

        x = np.array(x)
        y = np.array(y)
        label_dict = {i: [] for i in unary_collection}
        num_features = x.shape[1]
        y_classes = list(set(y))
        qsa_x = []
        for feature_index in range(num_features):
            if len(y_classes) > 2:
                for label in y_classes:
                    y_ = []
                    for i in y:
                        y_.append(1 if i == label else 0)
                    y_ = np.array(y_)
                    qsa_x.append(self.generate_qsa(x[:, feature_index], y_))

                    result_dict = self.valid_sample(x, y_, feature_index)
                    for op in unary_collection:
                        label_dict[op].append(result_dict[op])
            else:
                qsa_x.append(self.generate_qsa(x[:, feature_index], y))

                result_dict = self.valid_sample(x, y, feature_index)
                for op in unary_collection:
                    label_dict[op].append(result_dict[op])

        for key in label_dict:
            label_dict[key] = np.array(label_dict[key])

        qsa_x = np.array(qsa_x)
        with open(qsa_save_path, 'wb') as f:
            pkl.dump(qsa_x, f)
        with open(label_save_path, 'wb') as f:
            pkl.dump(label_dict, f)

        return qsa_x, label_dict

    def generate_qsa(self, x, y):  # Default one-vs-rest
        '''
        Convert a column into Quantile Sketch Array
        :param x: a column
        :param y: binary labels
        :return: Quantile Sketch Array
        '''
        scaler = MinMaxScaler(feature_range=(self.lower, self.upper))
        qsa = []
        for i in [0, 1]:
            x_ = [x[index] for index in range(len(x)) if y[index] == i]
            x_ = np.array(x_)
            x_ = np.reshape(x_, (len(x_), 1))
            x_ = scaler.fit_transform(x_)
            x_ = np.reshape(x_, (len(x_)))
            x_ -= self.lower
            bin_range = (self.upper - self.lower) / self.num_bins
            bucketized_col = np.zeros((self.num_bins,))
            for element in x_:
                index = int(element / bin_range)
                if index == self.num_bins:
                    index = self.num_bins - 1
                bucketized_col[index] += 1
            qsa.extend(bucketized_col / len(x_))
        return np.array(qsa)

    def fit(self, train_ops, data_dir='lfe/data', save_dir='lfe'):
        '''
        :param train_ops: list for train_ops
        :param data_dir: directory for training data
        :param save_dir: directory to save models
        :return:
        '''
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        train_x, train_y = self.load_training_data(data_dir)

        for train_op in train_ops:
            save_path = "lfe_" + self.name_prefix + "_" + train_op
            save_path = os.path.join(save_dir, save_path)
            if train_op == 'log':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            elif train_op == 'sqrt':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            elif train_op == 'square':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            elif train_op == 'freq':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            elif train_op == 'round':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            elif train_op == 'tanh':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            elif train_op == 'sigmoid':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            elif train_op == 'isoreg':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            elif train_op == 'zscore':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            elif train_op == 'norm':
                clf = MLP(hidden_layer_sizes=(500,), max_iter=3000, verbose=1, n_iter_no_change=20, tol=1e-5)
            else:
                raise ValueError("Unexpected operation %s" % train_op)
            clf.fit(train_x, train_y[train_op])
            from sklearn.metrics import accuracy_score
            print(accuracy_score(clf.predict(train_x), train_y[train_op]))
            with open(save_path, 'wb') as f:
                pkl.dump(clf, f)

    def predict(self, pred_op, x, save_dir='lfe'):
        '''

        :param pred_op: name of a unary operation, as shown below
        :param x: Quantile Sketch Array
        :param save_dir:
        :return: predictions, indicating the expected performance of each transformation
        '''

        save_path = "lfe_" + self.name_prefix + "_" + pred_op
        save_path = os.path.join(save_dir, save_path)
        with open(save_path, 'rb') as f:
            clf = pkl.load(f)

        pred = clf.predict_proba(x)
        return [element[1] for element in pred]

    def choose(self, x, y, save_dir='lfe'):
        '''
        Choose transformations for features
        :param x: features
        :param y: labels
        :param save_dir:
        :return: Operator if prediction > gamma, else None
        '''
        transformation = []
        x = np.array(x)
        num_features = x.shape[1]
        qsa_features = [self.generate_qsa(x[:, i], y) for i in range(num_features)]
        qsa_features = np.array(qsa_features)
        pred_dict = {}
        for pred_op in unary_collection:
            pred_dict[pred_op] = self.predict(pred_op, qsa_features, save_dir)
        for i in range(num_features):
            max_performance = -1
            best_op = ''
            for pred_op in unary_collection:
                pred = pred_dict[pred_op][i]
                if pred > max_performance:
                    max_performance = pred
                    best_op = pred_op

            if max_performance > self.gamma:
                tran = best_op
            else:
                tran = None
            transformation.append(tran)

        return transformation

    def valid_sample(self, x, y, t_id):
        '''
        Determine whether the t-th feature in features is a positive training sample
        :param x: original features
        :param y: ground truth label
        :param t_id: index of feature to be transformed
        :param threshold: threshold of improvement of newly constructed feature
        :return: dictionary, like {'log':1, 'sigmoid':0} 1 for positive and 0 for not positive
        '''
        x = np.array(x)
        y = np.array(y)
        kfold = SKFold(n_splits=10)
        results_org = []
        results_new = {op: [] for op in unary_collection}

        for train_index, test_index in kfold.split(x, y):
            # Original feature
            rfc_org = RFC()
            rfc_org.fit(x[train_index, t_id:t_id + 1], y[train_index])
            pred_org = rfc_org.predict(x[test_index, t_id:t_id + 1])
            results_org.append(f1_score(y[test_index], pred_org))

            # Constructed feature
            for op in unary_collection:
                operator = op_dict[op]
                rfc_new = RFC()
                new_feature = operator.operate(x[train_index, t_id])
                new_feature = np.reshape(new_feature, (len(new_feature), 1))
                rfc_new.fit(new_feature, y[train_index])
                # print(op,Counter(list(x[test_index, t_id])))
                new_feature = operator.operate(x[test_index, t_id])
                # print(op,Counter(list(new_feature)))
                new_feature = np.reshape(new_feature, (len(new_feature), 1))
                pred_new = rfc_new.predict(new_feature)
                results_new[op].append(f1_score(y[test_index], pred_new))

        result_org = np.mean(results_org)
        result_dict = {}
        for key in results_new:
            result_new = np.mean(results_new[key])
            if result_new >= result_org * (1 + self.theta):
                result_dict[key] = 1
            else:
                result_dict[key] = 0

        return result_dict

    def load_training_data(self, data_dir='lfe/data'):
        data = {}
        for root, _, files in os.walk(data_dir):
            for file in files:
                path = os.path.join(root, file)
                dataset = '_'.join(file.split('_')[1:])
                if dataset in exclude_dataset:
                    continue
                with open(path, 'rb') as f:
                    if file.split('_')[0] == 'qsa':
                        qsa = pkl.load(f)
                        if dataset not in data:
                            data[dataset] = {'qsa': qsa}
                        else:
                            data[dataset]['qsa'] = qsa
                    elif file.split('_')[0] == 'label':
                        label_dict = pkl.load(f)
                        if dataset not in data:
                            data[dataset] = {'label': label_dict}
                        else:
                            data[dataset]['label'] = label_dict
        train_x = []
        train_y = {op: [] for op in unary_collection}
        for key in data:
            train_x.extend(data[key]['qsa'])
            for op in unary_collection:
                train_y[op].extend(data[key]['label'][op])
        for op in unary_collection:
            train_y[op] = np.array(train_y[op])
        train_x = np.array(train_x)
        return train_x, train_y


def oversample(x, y, sample_size=0.2):
    num_samples = len(x)
    num_oversamples = int(sample_size * num_samples)
    print(y)
    true_inx = [i for i in range(num_samples) if y[i] == 1]
    oversample_idx = np.random.choice(true_inx, num_oversamples)
    _x = np.vstack((x, x[oversample_idx]))
    _y = list(y[:])
    _y.extend(list(y[oversample_idx]))
    return _x, np.array(_y)
