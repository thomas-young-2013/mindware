import os
import sys

from torchvision import transforms
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

sys.path.append(os.getcwd())
from solnml.datasets.image_dataset import ImageDataset
from solnml.components.models.img_classification.resnet50 import ResNet50Classifier
from solnml.components.utils.mfse_utils.config_space_utils import sample_configurations
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space
from solnml.components.evaluators.dl_evaluator import DLEvaluator
from solnml.components.metrics.metric import get_metric
from solnml.components.utils.constants import IMG_CLS


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
evaluator = DLEvaluator(None,
                        IMG_CLS,
                        scorer=get_metric('acc'),
                        dataset=image_data,
                        device='cuda',
                        image_size=32)


from solnml.components.computation.parallel_process import ParallelProcessEvaluator
config_space = ResNet50Classifier.get_hyperparameter_search_space()
model = UnParametrizedHyperparameter("estimator", 'resnet50')
config_space.add_hyperparameter(model)
aug_space = get_aug_hyperparameter_space()
config_space.add_hyperparameters(aug_space.get_hyperparameters())
config_space.add_conditions(aug_space.get_conditions())

executor = ParallelProcessEvaluator(evaluator, n_worker=3)
_configs = sample_configurations(config_space, 12)

res = executor.parallel_execute(_configs, resource_ratio=0.1)
print(res)
