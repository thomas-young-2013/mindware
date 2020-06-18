from sklearn.preprocessing import OneHotEncoder
import os
import sys
import torch
from torch import nn
from torchvision import transforms
import pickle as pkl

sys.path.append(os.getcwd())

from solnml.components.models.img_classification.resnext import ResNeXtClassifier
from solnml.components.models.img_classification.senet import SENetClassifier
from solnml.components.models.img_classification.nasnet import NASNetClassifier
from solnml.components.models.img_classification.nn_utils.dataset import get_array_dataset, get_folder_dataset
from solnml.components.utils.constants import MULTICLASS_CLS

from solnml.datasets.utils import load_data

# config = SENetClassifier.get_hyperparameter_search_space().sample_configuration().get_dictionary()
# config['epoch_num'] = 1
# config['batch_size'] = 48
# clf = SENetClassifier(**config, grayscale=True)
# x, y, _ = load_data('mnist_784', task_type=MULTICLASS_CLS, preprocess=False)
# ohe = OneHotEncoder(n_values=len(set(y)))
# with open('tmp.pkl', 'rb') as f:
#     x = pkl.load(f)
# print('ok')
# sample = 48
# train_dataset = get_array_dataset(x[:sample], y[:sample])
# val_dataset = get_array_dataset(x[-sample:], y[-sample:])
# clf.fit(train_dataset)
# print(clf.predict(val_dataset, 24))
# print(clf.predict_proba(val_dataset, 24))

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(560),
        transforms.RandomCrop(331),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(560),
        transforms.CenterCrop(331),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = 'data/img_datasets/hymenoptera_data'
train_dataset = get_folder_dataset(os.path.join(data_dir, 'train'), udf_transforms=data_transforms['train'])
val_dataset = get_folder_dataset(os.path.join(data_dir, 'val'), udf_transforms=data_transforms['val'])
config = ResNeXtClassifier.get_hyperparameter_search_space().sample_configuration().get_dictionary()
config['epoch_num'] = 1
clf = ResNeXtClassifier(**config)
clf.fit(train_dataset)
print(clf.predict(val_dataset, batch_size=48))
