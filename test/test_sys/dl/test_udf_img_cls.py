import os
import sys
from torchvision import transforms
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from solnml.datasets.image_dataset import ImageDataset
from solnml.estimators import ImageClassifier
from solnml.components.models.img_classification import add_classifier
from solnml.components.models.img_classification.resnext import ResNeXtClassifier

# Add user-defined classifier
add_classifier(ResNeXtClassifier)

data_dir = 'data/img_datasets/hymenoptera_data/'
image_data = ImageDataset(data_path=data_dir)
clf = ImageClassifier(time_limit=10,
                      include_algorithms=['ResNeXtClassifier'],
                      ensemble_method='ensemble_selection')
clf.fit(image_data)
image_data.load_test_data(data_dir)
print(clf.predict_proba(image_data))
print(clf.predict(image_data))
