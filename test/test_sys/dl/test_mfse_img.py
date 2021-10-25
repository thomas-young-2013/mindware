import os
import sys
import shutil
from torchvision import transforms
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from mindware.datasets.image_dataset import ImageDataset
from mindware.estimators import ImageClassifier
from mindware.components.models.img_classification.resnet50 import ResNet50Classifier
from mindware.components.models.img_classification.resnext import ResNeXtClassifier
from mindware.components.models.img_classification.senet import SENetClassifier
from mindware.components.models.img_classification.nasnet import NASNetClassifier

data_dir = 'data/img_datasets/extremely_small/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True)
save_dir = './data/eval_exps/mindware'
clf = ImageClassifier(time_limit=20,
                      max_epoch=3,
                      include_algorithms=['resnet32_32'],
                      ensemble_method='ensemble_selection',
                      evaluation='partial',
                      skip_profile=True,
                      output_dir=save_dir)
clf.fit(image_data, opt_method='whatever')
image_data.set_test_path(data_dir)
print(clf.predict_proba(image_data))
print(clf.predict(image_data))

# shutil.rmtree(save_dir)
