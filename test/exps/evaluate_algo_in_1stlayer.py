import os
import sys
import time
import pickle
import tabulate
import argparse
import numpy as np
sys.path.append(os.getcwd())
from automlToolkit.datasets.utils import load_data
from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.utils.functions import get_increasing_sequence

parser = argparse.ArgumentParser()
dataset_set = 'yeast,vehicle,diabetes,spectf,credit,' \
              'ionosphere,lymphography,messidor_features,winequality_red'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--methods', type=str, default='explore_first,exp3')
parser.add_argument('--mode', type=str, choices=['plot', 'exp'], default='exp')
parser.add_argument('--algo_num', type=int, default=8)
parser.add_argument('--trial_num', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep_num', type=int, default=10)

project_dir = './'
per_run_time_limit = 150


def evaluate_1stlayer_bandit(run_id, opt_algo, algorithms, dataset='credit', trial_num=200, seed=1):
    _start_time = time.time()
    raw_data = load_data(dataset, datanode_returned=True)
    bandit = FirstLayerBandit(trial_num, algorithms, raw_data,
                              output_dir='logs',
                              per_run_time_limit=per_run_time_limit,
                              dataset_name=dataset,
                              eval_type='holdout',
                              seed=seed)
    bandit.optimize(strategy=opt_algo)
    print(bandit.final_rewards)
    print(bandit.action_sequence)
    time_cost = time.time() - _start_time

    save_folder = project_dir + 'data/1stlayer-mab/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder + 'eval_1st_mab_%s_%s_%d_%d_%d.pkl' % (
        opt_algo, dataset, run_id, trial_num, len(algorithms))
    with open(save_path, 'wb') as f:
        data = [bandit.final_rewards, bandit.time_records, bandit.action_sequence, time_cost]
        pickle.dump(data, f)

    return time_cost


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    methods = args.methods.split(',')
    start_id = args.start_id
    rep_num = args.rep_num
    mode = args.mode
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    algorithms = ['k_nearest_neighbors', 'libsvm_svc', 'random_forest', 'adaboost']
    if algo_num == 8:
        algorithms = ['lda', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                      'adaboost', 'random_forest', 'extra_trees', 'decision_tree']

    dataset_list = dataset_str.split(',')

    if mode == 'exp':
        for dataset in dataset_list:
            for _id in range(start_id, start_id + rep_num):
                for method in methods:
                    time_cost = evaluate_1stlayer_bandit(
                        _id, method, algorithms,
                        dataset=dataset,
                        trial_num=trial_num,
                        seed=seeds[_id]
                    )
    else:
        headers = ['dataset', 'explore_first_mu', 'explore_first_var', 'exp3_mu', 'exp3_var', 'ducb_mu', 'ducb_var']
        tbl_data = list()
        for dataset in dataset_list:
            row_data = [dataset]
            for mth in ['explore_first', 'exp3', 'discounted_ucb']:
                results = list()
                for run_id in range(rep_num):
                    save_folder = project_dir + 'data/1stlayer-mab/'
                    file_path = save_folder + 'eval_1st_mab_%s_%s_%d_%d_%d.pkl' % (
                        mth, dataset, run_id, trial_num, len(algorithms))
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    final_rewards, action_sequence, evaluation_cost, _ = data
                    results.append(final_rewards)
                if len(results) == rep_num:
                    array = list()
                    for item in results:
                        item = get_increasing_sequence(item)
                        if len(item) < trial_num+1:
                            item.extend([item[-1]] * (1+trial_num - len(item)))
                        item = item[:trial_num+1]
                        assert len(item) == trial_num + 1
                        array.append(item)

                    mean_values = np.mean(array, axis=0)
                    std_value = np.std(np.asarray(array)[:, -1])
                    row_data.append('%.2f%%' % (100 * mean_values[-1]))
                    row_data.append('%.4f' % std_value)
                    print('=' * 30)
                    print('%s-%s: %.2f%%' % (dataset, mth, 100 * mean_values[-1]))
                    print('-' * 30)
                    print(mean_values)
                    print('=' * 30)
                else:
                    row_data.extend(['-', '-'])

            tbl_data.append(row_data)
        print(tabulate.tabulate(tbl_data, headers, tablefmt='github'))
