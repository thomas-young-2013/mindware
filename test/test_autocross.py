import sys

sys.path.append('/home/daim_gpu/sy/AlphaML')
sys.path.append('/home/daim_gpu/sy/Feature-Engineering')

from autocross import *
from alphaml.engine.components.data_manager import DataManager
from alphaml.datasets.cls_dataset.dataset_loader import load_data
from sklearn.model_selection import train_test_split

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='pc4')
args = parser.parse_args()


def test():
    ac = AutoCross(max_iter=10, metrics='acc')
    dataset_name = args.dataset_name
    x, y, _ = load_data(dataset_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75)
    dm = DataManager(x_train, y_train)
    feature_type = dm.info['feature_type']
    # categorical_index = [i for i in range(len(feature_type)) if feature_type[i] == 'Discrete']
    numerical_index = [i for i in range(len(feature_type)) if
                       feature_type[i] == 'Numerical' or feature_type[i] == 'Discrete']
    ac.fit(dm.train_X, dm.train_y, numerical_index=numerical_index)
    gen_feature = ac.transform(x_test)
    print(gen_feature)
    print(gen_feature.shape[1])


def test_transform():
    ac = AutoCross(max_iter=10, metrics='acc')
    dataset_name = args.dataset_name
    x, y, _ = load_data(dataset_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75)
    dm = DataManager(x_train, y_train)
    feature_type = dm.info['feature_type']
    # categorical_index = [i for i in range(len(feature_type)) if feature_type[i] == 'Discrete']
    numerical_index = [i for i in range(len(feature_type)) if
                       feature_type[i] == 'Numerical' or feature_type[i] == 'Discrete']
    ac.fit(dm.train_X, dm.train_y, numerical_index=numerical_index)
    gen_feature = ac.transform(x_test)
    print(gen_feature)
    print(gen_feature.shape[1])


test()
