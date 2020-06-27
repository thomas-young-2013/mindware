import os
import sys
from torchvision import transforms
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from solnml.datasets.image_dataset import ImageDataset
from solnml.estimators import ImageClassifier
from solnml.components.models.img_classification.resnet50 import ResNetClassifier
from solnml.components.models.img_classification.resnext import ResNeXtClassifier
from solnml.components.models.img_classification.senet import SENetClassifier
from solnml.components.models.img_classification.nasnet import NASNetClassifier

phase = 'fit'

if phase == 'fit':
    # data_dir = 'data/img_datasets/hymenoptera_data/'
    data_dir = 'data/img_datasets/extremely_small/'
    image_data = ImageDataset(data_path=data_dir)
    clf = ImageClassifier(time_limit=10,
                          include_algorithms=['resnext'],
                          ensemble_method='ensemble_selection')
    clf.fit(image_data)
    image_data.load_test_data(data_dir)
    print(clf.predict_proba(image_data))
    print(clf.predict(image_data))

else:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(560),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(560),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = 'data/img_datasets/hymenoptera_data/'
    image_data = ImageDataset(data_path=data_dir)
    image_data.set_udf_transform(data_transforms)
    image_data.load_data()
    image_data.load_test_data(data_dir)
    default_config = ResNeXtClassifier.get_hyperparameter_search_space().get_default_configuration().get_dictionary()
    default_config['device'] = 'cuda:0'
    default_config['epoch_num'] = 1
    print(default_config)
    clf = ResNeXtClassifier(**default_config)
    clf.fit(image_data)
    print(clf.score(image_data, accuracy_score, batch_size=16))
    image_data.val_dataset = image_data.train_dataset
    print(clf.score(image_data, accuracy_score, batch_size=16))
    print(clf.predict(image_data.test_dataset))
    print(clf.predict_proba(image_data.test_dataset))
