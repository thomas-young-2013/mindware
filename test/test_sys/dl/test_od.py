import os
import sys
import torch
from torch import nn
import pickle as pkl
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())
from solnml.components.models.object_detection.yolov3 import Yolov3
from solnml.datasets.od_dataset import ODDataset
from solnml.estimators import ObjectionDetecter

mode = 'fit'
if mode == 'fit':
    dataset = ODDataset('data/od_datasets/custom/custom.data')
    clf = ObjectionDetecter(time_limit=30,
                            include_algorithms=['yolov3'],
                            ensemble_method='ensemble_selection')
    clf.fit(dataset)

    dataset.set_test_path('data/od_datasets/custom/valid.txt')
    print(clf.predict(dataset))
else:
    config = Yolov3.get_hyperparameter_search_space().sample_configuration().get_dictionary()
    config['epoch_num'] = 1
    clf = Yolov3(**config)

    dataset = ODDataset('data/od_datasets/custom/custom.data')

    dataset.load_data()
    clf.fit(dataset)

    dataset.set_test_path('data/od_datasets/custom/valid.txt')
    dataset.load_test_data()
    print(clf.predict(dataset.test_dataset))
    print(clf.score(dataset))
