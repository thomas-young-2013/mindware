import os
import sys
import time
import pickle
import argparse
import tabulate
import numpy as np
import autosklearn.classification
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from automlToolkit.datasets.utils import load_train_test_data
from sklearn.model_selection import StratifiedShuffleSplit
from automlToolkit.components.feature_engineering.transformation_graph import DataNode, TransformationGraph


def train_valid_split(node: DataNode):
    X, y = node.copy_().data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    for train_index, test_index in sss.split(X, y):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
    train_data = DataNode(data=[X_train, y_train], feature_type=node.feature_types.copy())
    valid_data = DataNode(data=[X_val, y_val], feature_type=node.feature_types.copy())
    return train_data, valid_data


def train_valid_split_X(X, y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    for train_index, test_index in sss.split(X, y):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    trans_id = 6
    trans_types = ['fast_ica',
                   'quantile',
                   'variance_selector',
                   'percentile_selector',
                   'generic_selector',
                   'svd',
                   'feature_agg']
    trans_name = trans_types[trans_id]
    raw_data, _ = load_train_test_data('yeast')
    train_data, valid_data = train_valid_split(raw_data)

    X, y = raw_data.data
    if trans_name == 'fast_ica':
        from sklearn.decomposition import FastICA
        qt = FastICA(n_components=7, random_state=1)
    elif trans_name == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer()
    elif trans_name == 'variance_selector':
        from sklearn.feature_selection import VarianceThreshold
        qt = VarianceThreshold()
    elif trans_name == 'percentile_selector':
        from sklearn.feature_selection import SelectPercentile
        qt = SelectPercentile()
    elif trans_name == 'generic_selector':
        from sklearn.feature_selection import f_classif
        from sklearn.feature_selection import GenericUnivariateSelect
        qt = GenericUnivariateSelect(score_func=f_classif)
    elif trans_name == 'svd':
        from sklearn.decomposition import TruncatedSVD
        qt = TruncatedSVD(algorithm='randomized')
    elif trans_name == 'feature_agg':
        from sklearn.cluster import FeatureAgglomeration
        qt = FeatureAgglomeration()
    else:
        raise ValueError('Unsupported transformation name: %!' % trans_name)

    qt.fit(X, y)
    print(X.shape)

    # Case1: transform and split.
    x1 = qt.transform(X)
    _, x1_, _, _ = train_valid_split_X(x1, y)

    # Case2: split and transform.
    x2 = qt.transform(valid_data.data[0])
    # flag = np.isclose(x1_, x2)
    flag = (x1_ == x2)
    print(flag)
    print('='*50)
    for idx, item in enumerate(flag):
        if (item == False).any():
            print('Line - %d' % idx)
            print(item)
            print(x1_[idx])
            print(x2[idx])
    print('='*50)
    print(sum(flag))
    print('='*50)
    print('Transformation  :', trans_name)
    print('Is close        :', np.isclose(x1_, x2).all())
    print('Absolutely Same :', (x1_ == x2).all())
