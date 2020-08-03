import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from solnml.utils.functions import get_increasing_sequence

parser = argparse.ArgumentParser()
dataset_set = 'dna,pollen,abalone,splice,madelon,spambase,wind,page-blocks(1),pc2,segment'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--algo_num', type=int, default=15)
parser.add_argument('--time_limit', type=int, default=120)
parser.add_argument('--rep_num', type=int, default=10)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--ensemble', type=int, choices=[0, 1], default=1)

save_dir = './data/exp_results/exp1/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

args = parser.parse_args()
dataset_str = args.datasets
algo_num = args.algo_num
start_id = args.start_id
time_limit = args.time_limit
rep_num = args.rep_num
enable_ensemble = bool(args.ensemble)


def create_point(x, data):
    timestamps, perfs = data
    last_p = 0
    for t, p in zip(timestamps, perfs):
        if t > x:
            break
        last_p = p
    return last_p


def create_plot_points(data, start_time, end_time, point_num=500):
    x = np.linspace(start_time, end_time, num=point_num)
    result = list()
    for i, stage in enumerate(x):
        perf = create_point(stage, data)
        result.append(perf)
    return result


if __name__ == "__main__":

    dataset_list = dataset_str.split(',')

    ausk_id = 'ausk%d' % enable_ensemble
    method_ids = ['hmab', ausk_id]
    point_num = 500

    for dataset in dataset_list:
        x = np.linspace(0, time_limit, num=point_num)
        max_val = 0.
        for mth in method_ids:
            results = list()
            for run_id in range(rep_num):
                task_id = '[%s][%s-%d-%d]' % (mth, dataset, algo_num, time_limit)
                file_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
                if not os.path.exists(file_path):
                    continue
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                _, _, stats = data
                stats[1] = get_increasing_sequence(stats[1])
                max_val = np.max(stats[1])
                print('trial num', len(stats[0]))
                stats = create_plot_points(stats, 0, time_limit, point_num=point_num)
                results.append(stats)

            if len(results) == rep_num:
                results = np.array(results)
                means, stds = np.mean(results, axis=0), np.std(results, axis=0)
                max_val = np.max(means)
                label_name = 'auto-sklearn' if mth.startswith('ausk') else 'ours'
                plt.plot(x, means, label=label_name)

        pmin, pmax = max_val*0.8, max_val+0.03
        if pmax > 1.0:
            pmax = max_val + 0.01
        if pmax - pmin > 0.05:
            pmin = pmax - 0.05
        plt.ylim(pmin, pmax)
        plt.xlabel('Timestamps (s)')
        plt.ylabel('Validation Accuracy.')

        plt.title("Experiment 1 - %s" % dataset)
        plt.legend()
        plt.savefig('/Users/thomasyoung/Desktop/figures/exp1_%s_%d.pdf' % (dataset, time_limit))
        plt.show()
