import os
import sys
sys.path.append(os.getcwd())

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UnParametrizedHyperparameter

from solnml.components.utils.constants import IMG_CLS, TEXT_CLS, OBJECT_DET
from solnml.datasets.base_dl_dataset import DLDataset
from solnml.components.ensemble.dl_ensemble.ensemble_bulider import EnsembleBuilder, ensemble_list
from solnml.components.hpo_optimizer import build_hpo_optimizer
from solnml.components.evaluators.dl_evaluator import DLEvaluator
from solnml.components.evaluators.base_dl_evaluator import get_estimator_with_parameters, TopKModelSaver, get_estimator
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space, \
    get_test_transforms
from solnml.components.metrics.metric import get_metric
from solnml.datasets.image_dataset import ImageDataset
from solnml.estimators import ImageClassifier


def get_model_config_space(estimator_id, include_estimator=True, include_aug=True):
    from solnml.components.models.img_classification import _classifiers as _img_estimators, _addons as _img_addons
    clf_class = _img_estimators[estimator_id]
    default_cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", estimator_id)
    if include_estimator:
        default_cs.add_hyperparameter(model)
    if include_aug is True:
        aug_space = get_aug_hyperparameter_space()
        default_cs.add_hyperparameters(aug_space.get_hyperparameters())
        default_cs.add_conditions(aug_space.get_conditions())
    return default_cs


def get_pipeline_config_space(algorithm_candidates):
    cs = ConfigurationSpace()
    estimator_choice = CategoricalHyperparameter("estimator", algorithm_candidates,
                                                 default_value=algorithm_candidates[0])
    cs.add_hyperparameter(estimator_choice)
    aug_space = get_aug_hyperparameter_space()
    cs.add_hyperparameters(aug_space.get_hyperparameters())
    cs.add_conditions(aug_space.get_conditions())

    for estimator_id in algorithm_candidates:
        sub_cs = get_model_config_space(estimator_id, include_estimator=False, include_aug=False)
        parent_hyperparameter = {'parent': estimator_choice,
                                 'value': estimator_id}
        cs.add_configuration_space(estimator_id, sub_cs,
                                   parent_hyperparameter=parent_hyperparameter)
    return cs


cs = get_pipeline_config_space(['resnet34', 'mobilenet'])
dataset = 'cifar10'
data_dir = 'data/img_datasets/%s/' % dataset
image_data = ImageDataset(data_path=data_dir, train_val_split=True)

hpo_evaluator = DLEvaluator(cs.get_default_configuration(),
                            IMG_CLS,
                            scorer=get_metric('acc'),
                            dataset=image_data,
                            device='cuda',
                            image_size=32,
                            seed=1)
hpo_evaluator(cs.get_default_configuration())
