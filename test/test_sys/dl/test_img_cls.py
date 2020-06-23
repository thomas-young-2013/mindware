import os
import sys
from torchvision import transforms

sys.path.append(os.getcwd())

from solnml.datasets.image_dataset import ImageDataset
from solnml.estimators import ImageClassifier
from solnml.components.models.img_classification.resnet50 import ResNetClassifier
from solnml.components.models.img_classification.resnext import ResNeXtClassifier
from solnml.components.models.img_classification.senet import SENetClassifier

phase = 'fit'

if phase == 'fit':
    # data_dir = 'data/img_datasets/hymenoptera_data/'
    data_dir = 'data/img_datasets/extremely_small/'
    image_data = ImageDataset(data_path=data_dir)
    clf = ImageClassifier(time_limit=30, include_algorithms=['resnext'])
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
    # data_dir = 'data/img_datasets/hymenoptera_data/'
    data_dir = 'data/img_datasets/extremely_small/'
    image_data = ImageDataset(data_path=data_dir)
    image_data.set_udf_transform(data_transforms)
    image_data.load_data()
    image_data.load_test_data(data_dir)
    default_config = SENetClassifier.get_hyperparameter_search_space().get_default_configuration().get_dictionary()
    default_config['device'] = 'cuda:0'
    clf = SENetClassifier(**default_config)
    clf.fit(image_data)
    print(clf.predict(image_data.test_dataset))
    print(clf.predict_proba(image_data.test_dataset))
