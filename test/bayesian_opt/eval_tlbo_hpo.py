import os
import sys
import argparse

sys.path.append(os.getcwd())

from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator
from solnml.components.models.classification import _classifiers
from solnml.components.utils.constants import MULTICLASS_CLS

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='libsvm_svc')
parser.add_argument('--datasets', type=str, default='dna')
parser.add_argument('--n_jobs', type=int, default=1)

args = parser.parse_args()
test_datasets = args.datasets.split(',')
print(len(test_datasets))
algo_name = args.algo
max_runs = 70
rep = 10


def get_estimator(config):
    from solnml.components.models.classification import _classifiers, _addons
    classifier_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    config_['random_state'] = 1
    try:
        estimator = _classifiers[classifier_type](**config_)
    except:
        estimator = _addons.components[classifier_type](**config_)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', args.n_jobs)
    return classifier_type, estimator


def evaluate(dataset):
    train_data, test_data = load_train_test_data(dataset, test_size=0.3, task_type=MULTICLASS_CLS)

    cs = _classifiers[algo_name].get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", algo_name)
    cs.add_hyperparameter(model)
    default_hpo_config = cs.get_default_configuration()
    metric = get_metric('bal_acc')

    evaluator = ClassificationEvaluator(default_hpo_config, data_node=train_data,
                                        scorer=metric,
                                        name='hpo',
                                        resampling_strategy='holdout',
                                        seed=1)

    from solnml.components.hpo_optimizer.tlbo_optimizer import TlboOptimizer

    optimizer = TlboOptimizer(evaluator, cs, time_limit=300)
    optimizer.run()


def check_datasets(datasets, task_type=MULTICLASS_CLS):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1, task_type=task_type)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


check_datasets(test_datasets)
for dataset in test_datasets:
    evaluate(dataset)
