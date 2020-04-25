import os
import sys
import pickle
import argparse
import numpy as np
sys.path.append(os.getcwd())
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.components.utils.constants import MULTICLASS_CLS, BINARY_CLS
from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.metrics.metric import get_metric
from automlToolkit.components.evaluators.base_evaluator import fetch_predict_estimator

parser = argparse.ArgumentParser()
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep', type=int, default=3)
parser.add_argument('--datasets', type=str, default='diabetes')
parser.add_argument('--metrics', type=str, default='acc')
parser.add_argument('--task', type=str, choices=['reg', 'cls'], default='cls')
parser.add_argument('--algo', type=str, default='random_forest')
parser.add_argument('--r', type=int, default=20)
args = parser.parse_args()

datasets = args.datasets.split(',')
start_id, rep, algo = args.start_id, args.rep, args.algo
total_resource = args.r
save_dir = './data/meta_res/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
cls_metrics = ['acc', 'f1', 'auc']
reg_metrics = ['mse', 'r2', 'mae']


def evaluate_ml_algorithm(dataset, algo, run_id, obj_metric, total_resource=20, seed=1):
    print('EVALUATE-%s-%s: run_id=%d' % (dataset, algo, run_id))
    train_data, test_data = load_train_test_data(dataset)
    cls_task_type = BINARY_CLS if len(set(train_data.data[1])) == 2 else MULTICLASS_CLS
    print(set(train_data.data[1]))
    metric = get_metric(obj_metric)
    bandit = SecondLayerBandit(cls_task_type, algo, train_data, metric, per_run_time_limit=300,
                               seed=seed, eval_type='holdout',
                               fe_algo='bo',
                               total_resource=total_resource)
    bandit.optimize_fixed_pipeline()
    best_config = bandit.inc['hpo']

    fe_optimizer = bandit.optimizer['fe']
    fe_optimizer.fetch_nodes(10)
    best_data_node = bandit.inc['fe']
    test_data_node = fe_optimizer.apply(test_data, best_data_node)

    estimator = fetch_predict_estimator(cls_task_type, best_config, best_data_node.data[0],
                                        best_data_node.data[1],
                                        weight_balance=best_data_node.enable_balance,
                                        data_balance=best_data_node.data_balance)
    pred = estimator.predict(test_data_node.data[0])
    score = metric(test_data_node.data[1], pred)
    print('Test score', score)

    save_path = save_dir + '%s_%s_%s_%d_%d.pkl' % (dataset, algo, obj_metric, run_id, total_resource)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, algo, score, cls_task_type], f)


if __name__ == "__main__":
    algorithms = ['lightgbm', 'random_forest',
                  'libsvm_svc', 'extra_trees',
                  'liblinear_svc', 'k_nearest_neighbors',
                  'logistic_regression',
                  'gradient_boosting', 'adaboost']
    if args.task == 'reg':
        algorithms = ['lightgbm', 'random_forest',
                      'libsvm_svr', 'extra_trees',
                      'liblinear_svr', 'k_nearest_neighbors',
                      'lasso_regression',
                      'gradient_boosting', 'adaboost']
    metrics = args.metrics.split(',')

    for obj_metric in metrics:
        for dataset in datasets:
            np.random.seed(1)
            seeds = np.random.randint(low=1, high=10000, size=start_id + rep)
            for algo in algorithms:
                for run_id in range(start_id, start_id+rep):
                    seed = seeds[run_id]
                    evaluate_ml_algorithm(dataset, algo, run_id, obj_metric, total_resource=total_resource, seed=seed)
