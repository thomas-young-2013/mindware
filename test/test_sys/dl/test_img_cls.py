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
from solnml.components.models.img_classification.resnet34 import ResNet34Classifier
from solnml.components.models.img_classification.resnet101 import ResNet101Classifier
from solnml.components.models.img_classification.resnext import ResNeXtClassifier
from solnml.components.models.img_classification.senet import SENetClassifier
from solnml.components.models.img_classification.nasnet import NASNetClassifier
from solnml.components.models.img_classification.mobilenet import MobileNettClassifier
from solnml.components.models.img_classification.efficientnet import EfficientNetClassifier
from solnml.components.models.img_classification.resnet110_32 import ResNet110_32Classifier
from solnml.components.models.img_classification.densenet190_32 import DenseNet190_32Classifier
from solnml.components.models.img_classification.densenet100_32 import DenseNet100_32Classifier

phase = 'test'

if phase == 'fit':
    # data_dir = 'data/img_datasets/hymenoptera_data/'
    data_dir = 'data/img_datasets/cifar10/'
    # data_dir = 'data/img_datasets/dogs-vs-cats/'
    image_data = ImageDataset(data_path=data_dir, train_val_split=True, image_size=32)
    clf = ImageClassifier(time_limit=3600 * 10,
                          include_algorithms=['resnet44_32'],
                          evaluation='partial',
                          max_epoch=120,
                          skip_profile=True,
                          ensemble_method='ensemble_selection',
                          n_jobs=3)
    clf.fit(image_data, opt_method='whatever')
    image_data.set_test_path(data_dir)
    print(clf.score(image_data, mode='val'))
    print(clf.score(image_data))
    pred = clf.predict(image_data)
    timestamp = time.time()
    with open('es_output_%s.pkl' % timestamp, 'wb') as f:
        pkl.dump(pred, f)

else:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'val': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    }

    # data_dir = 'data/img_datasets/dogs-vs-cats/'
    data_dir = 'data/img_datasets/cifar10/'
    image_data = ImageDataset(data_path=data_dir, train_val_split=True, image_size=32, val_split_size=0.1)
    image_data.load_data(data_transforms['train'], data_transforms['val'])
    image_data.set_test_path(data_dir)

    # default_config = ResNet110_32Classifier.get_hyperparameter_search_space().sample_configuration().get_dictionary()
    default_config = {'device': 'cuda',
                      'epoch_num': 200,
                      'optimizer': 'SGD',
                      'sgd_learning_rate': 0.1,
                      'sgd_momentum': 0.9,
                      'nesterov': 'True',
                      'batch_size': 128,
                      'lr_decay': 0.1,
                      'weight_decay': 1e-4}
    clf = ResNet110_32Classifier(**default_config)
    clf.fit(image_data)
    # print(clf.score(image_data, accuracy_score, batch_size=16))
    # image_data.val_dataset = image_data.train_dataset
    # print(clf.score(image_data, accuracy_score, batch_size=16))
    # print(clf.predict(image_data.test_dataset))
    # print(clf.predict_proba(image_data.test_dataset))
