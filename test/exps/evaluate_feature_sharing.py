import os
import sys
import time
import pickle
import argparse
sys.path.append(os.getcwd())

from solnml.datasets.utils import load_data
from solnml.bandits.first_layer_bandit import FirstLayerBandit

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--algo_num', type=int, default=8)
parser.add_argument('--trial_num', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)


project_dir = './'
per_run_time_limit = 150


def evaluate_1stlayer_bandit(algorithms, mode, dataset='credit', trial_num=200, seed=1):
    _start_time = time.time()
    raw_data = load_data(dataset, datanode_returned=True)
    bandit = FirstLayerBandit(trial_num, algorithms, raw_data,
                              output_dir='logs',
                              per_run_time_limit=per_run_time_limit,
                              dataset_name=dataset,
                              share_feature=mode,
                              seed=seed)
    bandit.optimize()
    print(bandit.final_rewards)
    print(bandit.action_sequence)
    time_cost = time.time() - _start_time

    save_path = project_dir + 'data/shared_hmab_%d_%s_%d_%d_%d.pkl' % (
        mode, dataset, trial_num, len(algorithms), seed)
    with open(save_path, 'wb') as f:
        data = [bandit.final_rewards, bandit.time_records, bandit.action_sequence, time_cost]
        pickle.dump(data, f)

    return time_cost


def benchmark(dataset, algo_num, trial_num, seed):
    algorithms = ['k_nearest_neighbors', 'libsvm_svc', 'random_forest', 'adaboost']
    if algo_num == 8:
        algorithms = ['lda', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                      'adaboost', 'random_forest', 'extra_trees', 'decision_tree']
    # algorithms = ['libsvm_svc', 'k_nearest_neighbors']
    time_cost = evaluate_1stlayer_bandit(algorithms, False, dataset,
                                         trial_num=trial_num, seed=seed)
    time_cost = evaluate_1stlayer_bandit(algorithms, True, dataset,
                                         trial_num=trial_num, seed=seed)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    seed = args.seed

    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set
    else:
        dataset_list = dataset_str.split(',')

    for dataset in dataset_list:
        benchmark(dataset, algo_num, trial_num, seed)
