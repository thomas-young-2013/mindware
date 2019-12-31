import os
import sys
import time
import pickle
import argparse
import numpy as np
import autosklearn.classification
sys.path.append(os.getcwd())

from automlToolkit.datasets.utils import load_data, load_train_test_data
from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--mode', type=str, choices=['ausk', 'mab', 'benchmark'], default='benchmark')
parser.add_argument('--algo_num', type=int, default=8)
parser.add_argument('--time_cost', type=int, default=1200)
parser.add_argument('--trial_num', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)


project_dir = './'
per_run_time_limit = 150


def evaluate_1stlayer_bandit(algorithms, run_id, dataset='credit', trial_num=200, seed=1):
    _start_time = time.time()
    raw_data, test_raw_data = load_train_test_data(dataset)
    bandit = FirstLayerBandit(trial_num, algorithms, raw_data,
                              output_dir='logs',
                              per_run_time_limit=per_run_time_limit,
                              dataset_name=dataset,
                              seed=seed,
                              eval_type='holdout')
    bandit.optimize()
    print(bandit.final_rewards)
    print(bandit.action_sequence)

    print(bandit.predict(test_raw_data).shape)
    print(dataset, bandit.score(test_raw_data))
    time_cost = time.time() - _start_time
    return time_cost


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    mode = args.mode
    seed = args.seed

    algorithms = ['k_nearest_neighbors', 'libsvm_svc', 'random_forest', 'adaboost']
    if algo_num == 8:
        algorithms = ['lda', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                      'adaboost', 'random_forest', 'extra_trees', 'decision_tree']

    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set
    else:
        dataset_list = dataset_str.split(',')

    for dataset in dataset_list:
        time_cost = evaluate_1stlayer_bandit(
            algorithms, 0, dataset, trial_num=trial_num, seed=seed
        )
