import os
import sys
import torch
from torch import nn
from torchvision import transforms
import pickle as pkl

sys.path.append(os.getcwd())
from solnml.components.models.text_classification.NaiveBert import NaiveBertClassifier
from solnml.components.models.text_classification.DPCNNBert import DPCNNBertClassifier
from solnml.components.models.text_classification.RCNNBert import RCNNBertClassifier
from solnml.components.models.text_classification.nn_utils.dataset import TextBertDataset

config = RCNNBertClassifier.get_hyperparameter_search_space().sample_configuration().get_dictionary()
config['epoch_num'] = 1
clf = RCNNBertClassifier(**config)
train_dataset = TextBertDataset('train.csv',
                                config_path='./solnml/components/models/text_classification/nn_utils/bert-base-uncased')
test_dataset = TextBertDataset('test.csv',
                               config_path='./solnml/components/models/text_classification/nn_utils/bert-base-uncased')
clf.fit(train_dataset)
print(clf.predict(test_dataset))
print(clf.predict_proba(test_dataset))
