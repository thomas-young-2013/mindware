import os
import sys
import pickle as pkl
import argparse
import numpy as np

sys.path.append(os.getcwd())
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.components.fe_optimizers.bo_optimizer import BayesianOptimizationOptimizer
from solnml.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from solnml.components.utils.constants import CLASSIFICATION, REGRESSION
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.evaluators.base_evaluator import fetch_predict_estimator
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator
from solnml.components.evaluators.rgs_evaluator import RegressionEvaluator

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='diabetes')
parser.add_argument('--metrics', type=str, default='acc')
parser.add_argument('--task', type=str, choices=['reg', 'cls'], default='cls')
parser.add_argument('--output_dir', type=str, default='./data/fe_hpo_results')
args = parser.parse_args()

dataset_list = args.datasets.split(',')
metric = get_metric(args.metrics)
algorithms = ['lightgbm', 'random_forest',
              'libsvm_svc', 'extra_trees',
              'liblinear_svc', 'k_nearest_neighbors',
              'logistic_regression',
              'gradient_boosting', 'adaboost']
task = args.task
if task == 'cls':
    from solnml.components.models.classification import _classifiers

    _estimators = _classifiers
else:
    from solnml.components.models.regression import _regressors

    _estimators = _regressors

eval_type = 'holdout'
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for dataset in dataset_list:
    train_data, test_data = load_train_test_data(dataset)
    for algo in algorithms:
        cs = _estimators[algo].get_hyperparameter_search_space()
        model = UnParametrizedHyperparameter("estimator", algo)
        cs.add_hyperparameter(model)
        default_hpo_config = cs.get_default_configuration()

        if task == 'cls':
            fe_evaluator = ClassificationEvaluator(default_hpo_config, scorer=metric,
                                                   name='fe', resampling_strategy=eval_type,
                                                   seed=1)
            hpo_evaluator = ClassificationEvaluator(default_hpo_config, scorer=metric,
                                                    data_node=train_data, name='hpo',
                                                    resampling_strategy=eval_type,
                                                    seed=1)
        else:
            fe_evaluator = RegressionEvaluator(default_hpo_config, scorer=metric,
                                               name='fe', resampling_strategy=eval_type,
                                               seed=1)
            hpo_evaluator = RegressionEvaluator(default_hpo_config, scorer=metric,
                                                data_node=train_data, name='hpo',
                                                resampling_strategy=eval_type,
                                                seed=1)

        fe_optimizer = BayesianOptimizationOptimizer(task_type=CLASSIFICATION if task == 'cls' else REGRESSION,
                                                     input_data=train_data,
                                                     evaluator=fe_evaluator,
                                                     model_id=algo,
                                                     time_limit_per_trans=600,
                                                     mem_limit_per_trans=5120,
                                                     number_of_unit_resource=10,
                                                     seed=1)
        hpo_optimizer = SMACOptimizer(evaluator=hpo_evaluator,
                                      config_space=cs,
                                      per_run_time_limit=600,
                                      per_run_mem_limit=5120,
                                      output_dir='./logs',
                                      trials_per_iter=100)
        fe_optimizer.iterate()
        fe_eval_dict = fe_optimizer.eval_dict
        fe_dict = {}
        for key, value in fe_eval_dict.items():
            fe_dict[key[0]] = value
        hpo_optimizer.iterate()
        hpo_eval_dict = hpo_optimizer.eval_dict
        hpo_dict = {}
        for key, value in hpo_eval_dict.items():
            hpo_dict[key[1]] = value
        with open(os.path.join(output_dir, '%s-%s-fe.pkl' % (dataset, algo)), 'wb') as f:
            pkl.dump(fe_dict, f)
        with open(os.path.join(output_dir, '%s-%s-hpo.pkl' % (dataset, algo)), 'wb') as f:
            pkl.dump(hpo_dict, f)

        print("Algo %s end" % algo)
