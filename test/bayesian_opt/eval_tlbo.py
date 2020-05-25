import os
import re
import sys
import pickle as pk

sys.path.append(os.getcwd())

from solnml.components.transfer_learning.tlbo.tlbo_optimizer import TLBO
from solnml.components.transfer_learning.tlbo.bo_optimizer import BO
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.components.fe_optimizers.bo_optimizer import BayesianOptimizationOptimizer
from solnml.components.utils.constants import CLASSIFICATION, REGRESSION
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator
from solnml.components.models.classification import _classifiers

task_id = 'fe'
algo_name = 'random_forest'
metric = 'acc'
datasets = list()

pattern = r'(.*)-%s-%s-%d-%s.pkl' % (algo_name, metric, 0, task_id)
data_dir = 'test/bayesian_opt/runhistory/config_res/'
for filename in os.listdir(data_dir):
    result = re.search(pattern, filename, re.M | re.I)
    if result is not None:
        datasets.append(result.group(1))
print(datasets)


def load_runhistory(dataset_names):
    runhistory = list()
    for dataset in dataset_names:
        _filename = '%s-%s-%s-%d-%s.pkl' % (dataset, 'random_forest', 'acc', 0, task_id)
        with open(data_dir + _filename, 'rb') as f:
            data = pk.load(f)
        runhistory.append(list(data.items()))
    return runhistory


dataset = datasets[0]
past_datasets = datasets[1:6]
print(past_datasets)
past_history = load_runhistory(past_datasets)

metric = get_metric(metric)
train_data, test_data = load_train_test_data(dataset)

cs = _classifiers[algo_name].get_hyperparameter_search_space()
model = UnParametrizedHyperparameter("estimator", algo_name)
cs.add_hyperparameter(model)
default_hpo_config = cs.get_default_configuration()
fe_evaluator = ClassificationEvaluator(default_hpo_config, scorer=metric,
                                       name='fe', resampling_strategy='holdout',
                                       seed=1)
fe_optimizer = BayesianOptimizationOptimizer(task_type=CLASSIFICATION,
                                             input_data=train_data,
                                             evaluator=fe_evaluator,
                                             model_id=algo_name,
                                             time_limit_per_trans=600,
                                             mem_limit_per_trans=5120,
                                             number_of_unit_resource=10,
                                             seed=1)
hyper_space = fe_optimizer.hyperparameter_space


def objective_function(config):
    return fe_optimizer.evaluate_function(config)


bo = BO(objective_function, hyper_space, max_runs=30)
# bo = TLBO(objective_function, hyper_space, past_history, max_runs=30)

bo.run()
print(bo.get_incumbent())
