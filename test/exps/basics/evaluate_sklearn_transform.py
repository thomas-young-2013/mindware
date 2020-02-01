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
    trans_id = 20
    trans_types = ['fast_ica',
                   'quantile',
                   'variance_selector',
                   'percentile_selector',
                   'generic_selector',
                   'svd',
                   'feature_agg',
                   'extra_tree_selector',
                   'liblinear_based_selector',
                   'rfe_selector',
                   'normalizer',
                   'scaler1',
                   'scaler2',
                   'scaler3',
                   'random_tree_embedding',
                   'polynomial',
                   'pca',
                   'nystronem',
                   'lda',
                   'kitchen_sink',
                   'kernel_pca',
                   'cross']
    trans_name = trans_types[trans_id]
    raw_data, _ = load_train_test_data('yeast')
    train_data, valid_data = train_valid_split(raw_data)

    X, y = raw_data.data
    if trans_name == 'fast_ica':
        from sklearn.decomposition import FastICA

        qt = FastICA(n_components=7, random_state=1)
    elif trans_name == 'quantile':
        from automlToolkit.components.feature_engineering.transformations.utils import QuantileTransformer

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
    elif trans_name == 'extra_tree_selector':
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier()
        qt = SelectFromModel(model)
    elif trans_name == 'liblinear_based_selector':
        from sklearn.feature_selection import SelectFromModel
        from sklearn.svm import LinearSVC

        model = LinearSVC()
        qt = SelectFromModel(model)
    elif trans_name == 'rfe_selector':
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        qt = RFECV(LogisticRegression())
    elif trans_name == 'normalizer':
        from sklearn.preprocessing import Normalizer

        qt = Normalizer()
    elif trans_name == 'scaler1':
        from sklearn.preprocessing import MinMaxScaler

        qt = MinMaxScaler()
    elif trans_name == 'scaler2':
        from sklearn.preprocessing import StandardScaler

        qt = StandardScaler()
    elif trans_name == 'scaler3':
        from sklearn.preprocessing import RobustScaler

        qt = RobustScaler()
    elif trans_name == 'random_tree_embedding':
        from sklearn.ensemble import RandomTreesEmbedding

        qt = RandomTreesEmbedding()
    elif trans_name == 'polynomial':
        from sklearn.preprocessing import PolynomialFeatures

        qt = PolynomialFeatures()
    elif trans_name == 'pca':
        from sklearn.decomposition import PCA

        qt = PCA()
    elif trans_name == 'nystronem':
        from sklearn.kernel_approximation import Nystroem

        qt = Nystroem()
    elif trans_name == 'kernel_pca':
        from automlToolkit.components.feature_engineering.transformations.utils import KernelPCA

        qt = KernelPCA()
    elif trans_name == 'kitchen_sink':
        from sklearn.kernel_approximation import RBFSampler

        qt = RBFSampler()
    elif trans_name == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        qt = LinearDiscriminantAnalysis()
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
    print('=' * 50)
    for idx, item in enumerate(flag):
        if (item == False).any():
            print('Line - %d' % idx)
            print(item)
            print(x1_[idx])
            print(x2[idx])
    print('=' * 50)
    print(sum(flag))
    print('=' * 50)
    print('Transformation  :', trans_name)
    print('Is close        :', np.isclose(x1_, x2).all())
    print('Absolutely Same :', (x1_ == x2).all())
