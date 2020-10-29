import os
import sys
import argparse
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

sys.path.append(os.getcwd())
from solnml.datasets.image_dataset import ImageDataset
from solnml.components.optimizers.base.config_space_utils import sample_configurations
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space
from solnml.components.evaluators.dl_evaluator import DLEvaluator
from solnml.components.metrics.metric import get_metric
from solnml.components.utils.constants import IMG_CLS

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='mobilenet')
parser.add_argument('--dataset', type=str, default='extremely_small')
args = parser.parse_args()

data_dir = 'data/img_datasets/%s/' % args.dataset

image_data = ImageDataset(data_path=data_dir, train_val_split=True)
image_data.set_test_path(data_dir)
evaluator = DLEvaluator(None,
                        IMG_CLS,
                        scorer=get_metric('acc'),
                        dataset=image_data,
                        device='cuda',
                        image_size=32)


from solnml.components.computation.parallel_process import ParallelProcessEvaluator
from solnml.components.models.img_classification import _classifiers
network_id = args.network
config_space = _classifiers[network_id].get_hyperparameter_search_space()
model = UnParametrizedHyperparameter("estimator", network_id)
config_space.add_hyperparameter(model)
aug_space = get_aug_hyperparameter_space()
config_space.add_hyperparameters(aug_space.get_hyperparameters())
config_space.add_conditions(aug_space.get_conditions())

executor = ParallelProcessEvaluator(evaluator, n_worker=3)
_configs = sample_configurations(config_space, 12)

res = executor.parallel_execute(_configs, resource_ratio=0.1)
print(res)
