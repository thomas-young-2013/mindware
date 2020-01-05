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

from automlToolkit.datasets.utils import load_data, load_train_test_data
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
    import numpy as np
    from sklearn.preprocessing import QuantileTransformer

    raw_data, _ = load_train_test_data('yeast')
    train_data, valid_data = train_valid_split(raw_data)

    X, y = raw_data.data
    qt = QuantileTransformer()
    qt.fit(X, y)
    print(X.shape)

    # Case1: transform and split.
    x1 = qt.transform(X)
    _, x1_, _, _ = train_valid_split_X(x1, y)

    # Case2: split and transform.
    x2 = qt.transform(valid_data.data[0])

    print(x1_ == x2)
    print((x1_ == x2).all())
