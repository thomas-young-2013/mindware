import os
import sys
from torchvision import transforms
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from solnml.datasets.image_dataset import ImageDataset
from solnml.estimators import ImageClassifier
from solnml.components.models.img_classification.resnet50 import ResNet50Classifier
from solnml.components.models.img_classification.resnext import ResNeXtClassifier
from solnml.components.models.img_classification.senet import SENetClassifier
from solnml.components.models.img_classification.nasnet import NASNetClassifier

data_dir = 'data/img_datasets/hymenoptera_data/'
image_data = ImageDataset(data_path=data_dir)
clf = ImageClassifier(time_limit=30,
                      include_algorithms=['resnext'],
                      ensemble_method='ensemble_selection',
                      evaluation='partial')
clf.fit(image_data)
image_data.set_test_path(data_dir)
print(clf.predict_proba(image_data))
print(clf.predict(image_data))
