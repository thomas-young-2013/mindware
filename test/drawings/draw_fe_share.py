import os
import sys
import pickle
import argparse
import numpy as np
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'

parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--algo_num', type=int, default=8)
parser.add_argument('--trial_num', type=int, default=100)

project_dir = './'


def plot(mode, dataset, algo_num, trial_num, seed):
    save_path = project_dir + 'data/shared_hmab_%d_%s_%d_%d_%d.pkl' % (
            mode, dataset, trial_num, algo_num, seed)
    with open(save_path, 'rb') as f:
        result = pickle.load(f)
    print('Best validation accuracy: %.4f' % np.max(result[0]))
    print('Final Rewards', result[0])
    print('Time records', result[1])
    print('Action Sequence', result[2])
    print('-' * 30)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set
    else:
        dataset_list = dataset_str.split(',')

    for dataset in dataset_list:
        plot(False, dataset, args.algo_num, args.trial_num, args.seed)
        plot(True, dataset, args.algo_num, args.trial_num, args.seed)


