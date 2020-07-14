import os
import sys
import time
import pickle as pkl
from torchvision import transforms
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from solnml.datasets.image_dataset import ImageDataset
from solnml.estimators import ImageClassifier
from solnml.components.models.img_classification.resnet50 import ResNet50Classifier
from solnml.components.models.img_classification.resnext import ResNeXtClassifier
from solnml.components.models.img_classification.senet import SENetClassifier
from solnml.components.models.img_classification.nasnet import NASNetClassifier
from solnml.components.models.img_classification.mobilenet import MobileNettClassifier
from solnml.components.models.img_classification.efficientnet import EfficientNetClassifier

phase = 'fit'

if phase == 'fit':
    # data_dir = 'data/img_datasets/hymenoptera_data/'
    data_dir = 'data/img_datasets/cifar10/'
    image_data = ImageDataset(data_path=data_dir, train_val_split=True)
    clf = ImageClassifier(time_limit=1800,
                          include_algorithms=['mobilenet'],
                          evaluation='holdout',
                          image_size=32,
                          max_epoch=30,
                          skip_profile=True,
                          ensemble_method='ensemble_selection',
                          n_jobs=3)
    clf.fit(image_data)
    image_data.set_test_path(data_dir)
    print(clf.score(image_data, mode='val'))
    print(clf.score(image_data))
    pred = clf.predict(image_data)
    timestamp = time.time()
    with open('es_output_%s.pkl' % timestamp, 'wb') as f:
        pkl.dump(pred, f)

# else:
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(560),
#             transforms.RandomCrop(256),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(560),
#             transforms.CenterCrop(256),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
#     # data_dir = 'data/img_datasets/dogs-vs-cats/'
#     data_dir = 'data/img_datasets/extremely_small/'
#     image_data = ImageDataset(data_path=data_dir, train_val_split=True)
#     image_data.load_data(data_transforms['train'], data_transforms['val'])
#     image_data.set_test_path(data_dir)
#     default_config = ResNet50Classifier.get_hyperparameter_search_space().get_default_configuration().get_dictionary()
#     default_config['device'] = 'cuda'
#     default_config['epoch_num'] = 150
#     print(default_config)
#     clf = ResNet50Classifier(**default_config)
#     clf.fit(image_data)
#     print(clf.score(image_data, accuracy_score, batch_size=16))
#     image_data.val_dataset = image_data.train_dataset
#     print(clf.score(image_data, accuracy_score, batch_size=16))
#     print(clf.predict(image_data.test_dataset))
#     print(clf.predict_proba(image_data.test_dataset))
