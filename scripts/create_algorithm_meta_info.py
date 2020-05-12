import os
import sys
import pickle
import argparse
import numpy as np

sys.path.append(os.getcwd())
from solnml.bandits.second_layer_bandit import SecondLayerBandit
from solnml.components.utils.constants import MULTICLASS_CLS, BINARY_CLS, REGRESSION, CLS_TASKS
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.evaluators.base_evaluator import fetch_predict_estimator

parser = argparse.ArgumentParser()
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep', type=int, default=3)
parser.add_argument('--datasets', type=str, default='diabetes')
parser.add_argument('--metrics', type=str, default='all')
parser.add_argument('--task', type=str, choices=['reg', 'cls'], default='cls')
parser.add_argument('--algo', type=str, default='all')
parser.add_argument('--r', type=int, default=20)
args = parser.parse_args()

datasets = args.datasets.split(',')
start_id, rep = args.start_id, args.rep
total_resource = args.r
save_dir = './data/meta_res/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
cls_metrics = ['acc', 'f1', 'auc']
reg_metrics = ['mse', 'r2', 'mae']


def evaluate_ml_algorithm(dataset, algo, run_id, obj_metric, total_resource=20, seed=1, task_type=None):
    print('EVALUATE-%s-%s-%s: run_id=%d' % (dataset, algo, obj_metric, run_id))
    train_data, test_data = load_train_test_data(dataset, task_type=task_type)
    if task_type in CLS_TASKS:
        task_type = BINARY_CLS if len(set(train_data.data[1])) == 2 else MULTICLASS_CLS
    print(set(train_data.data[1]))
    metric = get_metric(obj_metric)
    bandit = SecondLayerBandit(task_type, algo, train_data, metric, per_run_time_limit=300,
                               seed=seed, eval_type='holdout',
                               fe_algo='bo',
                               total_resource=total_resource)
    bandit.optimize_fixed_pipeline()

    val_score = bandit.incumbent_perf
    best_config = bandit.inc['hpo']

    fe_optimizer = bandit.optimizer['fe']
    fe_optimizer.fetch_nodes(10)
    best_data_node = fe_optimizer.incumbent
    test_data_node = fe_optimizer.apply(test_data, best_data_node)

    estimator = fetch_predict_estimator(task_type, best_config, best_data_node.data[0],
                                        best_data_node.data[1],
                                        weight_balance=best_data_node.enable_balance,
                                        data_balance=best_data_node.data_balance)
    score = metric(estimator, test_data_node.data[0], test_data_node.data[1]) * metric._sign
    print('Test score', score)

    save_path = save_dir + '%s-%s-%s-%d-%d.pkl' % (dataset, algo, obj_metric, run_id, total_resource)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, algo, score, val_score, task_type], f)


def check_datasets(datasets, task_type=None):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1, task_type=task_type)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == "__main__":
    algorithms = ['lightgbm', 'random_forest',
                  'libsvm_svc', 'extra_trees',
                  'liblinear_svc', 'k_nearest_neighbors',
                  'logistic_regression',
                  'gradient_boosting', 'adaboost']
    task_type = MULTICLASS_CLS
    if args.task == 'reg':
        task_type = REGRESSION
        algorithms = ['lightgbm', 'random_forest',
                      'libsvm_svr', 'extra_trees',
                      'liblinear_svr', 'k_nearest_neighbors',
                      'lasso_regression',
                      'gradient_boosting', 'adaboost']

    if args.algo != 'all':
        algorithms = args.algo.split(',')

    metrics = cls_metrics if args.task == 'cls' else reg_metrics
    if args.metrics != 'all':
        metrics = args.metrics.split(',')

    check_datasets(datasets, task_type=task_type)
    running_info = list()
    log_filename = 'running-%d.txt' % os.getpid()

    for dataset in datasets:
        for obj_metric in metrics:
            np.random.seed(1)
            seeds = np.random.randint(low=1, high=10000, size=start_id + rep)
            for algo in algorithms:
                for run_id in range(start_id, start_id + rep):
                    seed = seeds[run_id]
                    try:
                        task_id = '%s-%s-%s-%d: %s' % (dataset, algo, obj_metric, run_id, 'success')
                        evaluate_ml_algorithm(dataset, algo, run_id, obj_metric, total_resource=total_resource,
                                              seed=seed, task_type=task_type)
                    except Exception as e:
                        task_id = '%s-%s-%s-%d: %s' % (dataset, algo, obj_metric, run_id, str(e))

                    print(task_id)
                    running_info.append(task_id)
                    with open(save_dir + log_filename, 'a') as f:
                        f.write('\n' + task_id)

    # Write down the error info.
    with open(save_dir + 'failed-%s' % log_filename, 'w') as f:
        f.write('\n'.join(running_info))
