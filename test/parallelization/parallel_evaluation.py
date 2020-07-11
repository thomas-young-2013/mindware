import os
import sys
import time
import pickle as pkl
from torchvision import transforms
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())
from solnml.datasets.image_dataset import ImageDataset
from solnml.components.models.img_classification.resnet50 import ResNet50Classifier
from solnml.components.utils.mfse_utils.config_space_utils import sample_configurations


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
# data_dir = 'data/img_datasets/dogs-vs-cats/'
# data_dir = 'data/img_datasets/cifar10/'
data_dir = 'data/img_datasets/extremely_small/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True)
image_data.load_data(data_transforms['train'], data_transforms['val'])
image_data.set_test_path(data_dir)


def obj_function(image_data, config):
    pid = os.getpid()
    config = config.get_dictionary()
    config['device'] = 'cuda'
    config['epoch_num'] = 20
    print(pid, config)
    clf = ResNet50Classifier(**config)
    clf.fit(image_data)
    train_acc = clf.score(image_data, accuracy_score, batch_size=16)
    print(pid, 'training score', train_acc)
    return train_acc


from solnml.components.computation.parallel_process import ParallelProcessEvaluator
config_space = ResNet50Classifier.get_hyperparameter_search_space()
executor = ParallelProcessEvaluator(obj_function, n_worker=3)
_configs = sample_configurations(config_space, 12)
configs = [(image_data, _config) for _config in _configs]

executor.parallel_execute(configs)
