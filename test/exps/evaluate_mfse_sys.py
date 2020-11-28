import os
import sys
import time
import pickle
import argparse
import numpy as np
from tabulate import tabulate
from sklearn.metrics import make_scorer

sys.path.append(os.getcwd())

from solnml.datasets.utils import load_train_test_data
from solnml.bandits.first_layer_bandit import FirstLayerBandit
from solnml.components.metrics.cls_metrics import balanced_accuracy
from solnml.components.utils.constants import MULTICLASS_CLS, BINARY_CLS
from solnml.utils.functions import is_unbalanced_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='pc2')
parser.add_argument('--mode', type=str, choices=['ausk', 'hmab', 'hmab,ausk', 'plot'], default='plot')
parser.add_argument('--algo_num', type=int, default=15)
parser.add_argument('--time_cost', type=int, default=3600)
parser.add_argument('--rep_num', type=int, default=10)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

project_dir = './data/meta_exp/'
per_run_time_limit = 120
opt_algo = 'fixed'
hmab_flag = 'mfse'
ausk_flag = 'ausk'
if not os.path.exists(project_dir):
    os.makedirs(project_dir)


def evaluate_hmab(algorithms, dataset, run_id, trial_num, seed, time_limit=1200):
    print('%s-%s-%d: %d' % (hmab_flag, dataset, run_id, time_limit))

    _start_time = time.time()
    train_data, test_data = load_train_test_data(dataset, task_type=MULTICLASS_CLS)
    cls_task_type = BINARY_CLS if len(set(train_data.data[1])) == 2 else MULTICLASS_CLS
    balanced_acc_metric = make_scorer(balanced_accuracy)

    if is_unbalanced_dataset(train_data):
        from solnml.components.feature_engineering.transformations.balancer.smote_balancer import DataBalancer
        train_data = DataBalancer().operate(train_data)

    bandit = FirstLayerBandit(cls_task_type, trial_num, algorithms, train_data,
                              output_dir='logs',
                              per_run_time_limit=per_run_time_limit,
                              dataset_name=dataset,
                              ensemble_size=50,
                              inner_opt_algorithm=opt_algo,
                              metric=balanced_acc_metric,
                              fe_algo='bo',
                              seed=seed,
                              time_limit=time_limit,
                              eval_type='partial')
    while time.time() - _start_time < time_limit:
        bandit.sub_bandits['random_forest'].optimizer['hpo'].iterate()
    # bandit.optimize()
    # fe_exp_output = bandit.sub_bandits['random_forest'].exp_output['fe']
    # hpo_exp_output = bandit.sub_bandits['random_forest'].exp_output['hpo']
    fe_exp_output = dict()
    hpo_exp_output = bandit.sub_bandits['random_forest'].optimizer['hpo'].exp_output
    inc_config = bandit.sub_bandits['random_forest'].optimizer['hpo'].incumbent_config.get_dictionary()
    inc_config.pop('estimator')
    from solnml.components.models.classification.random_forest import RandomForest
    rf = RandomForest(**inc_config)
    rf.fit(train_data.data[0], train_data.data[1])
    validation_accuracy = bandit.sub_bandits['random_forest'].optimizer['hpo'].incumbent_perf
    best_pred = rf.predict(test_data.data[0])
    test_accuracy = balanced_accuracy(test_data.data[1], best_pred)

    # es_pred = bandit._es_predict(test_data)
    # test_accuracy_with_ens = balanced_accuracy(test_data.data[1], es_pred)
    data = [dataset, validation_accuracy, test_accuracy, fe_exp_output, hpo_exp_output,
            _start_time]
    save_path = project_dir + '%s_%s_%s_%d_%d_%d_%d_%d.pkl' % (
        hmab_flag, opt_algo, dataset, trial_num, len(algorithms), seed, run_id, time_limit)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    del_path = './logs/'
    for i in os.listdir(del_path):
        file_data = del_path + "/" + i
        if os.path.isfile(file_data):
            os.remove(file_data)


def evaluate_autosklearn(algorithms, dataset, run_id, trial_num, seed, time_limit=1200):
    print('%s-%s-%d: %d' % (hmab_flag, dataset, run_id, time_limit))

    _start_time = time.time()
    train_data, test_data = load_train_test_data(dataset, task_type=MULTICLASS_CLS)
    cls_task_type = BINARY_CLS if len(set(train_data.data[1])) == 2 else MULTICLASS_CLS
    balanced_acc_metric = make_scorer(balanced_accuracy)

    if is_unbalanced_dataset(train_data):
        from solnml.components.feature_engineering.transformations.balancer.smote_balancer import DataBalancer
        train_data = DataBalancer().operate(train_data)

    bandit = FirstLayerBandit(cls_task_type, trial_num, algorithms, train_data,
                              output_dir='logs',
                              per_run_time_limit=per_run_time_limit,
                              dataset_name=dataset,
                              ensemble_size=50,
                              inner_opt_algorithm=opt_algo,
                              metric=balanced_acc_metric,
                              fe_algo='bo',
                              seed=seed,
                              time_limit=time_limit,
                              eval_type='holdout')
    while time.time() - _start_time < time_limit:
        bandit.sub_bandits['random_forest'].optimizer['hpo'].iterate()
    # bandit.optimize()
    # fe_exp_output = bandit.sub_bandits['random_forest'].exp_output['fe']
    # hpo_exp_output = bandit.sub_bandits['random_forest'].exp_output['hpo']
    fe_exp_output = dict()
    hpo_exp_output = bandit.sub_bandits['random_forest'].optimizer['hpo'].exp_output
    inc_config = bandit.sub_bandits['random_forest'].optimizer['hpo'].incumbent_config.get_dictionary()
    inc_config.pop('estimator')
    from solnml.components.models.classification.random_forest import RandomForest
    rf = RandomForest(**inc_config)
    rf.fit(train_data.data[0], train_data.data[1])
    validation_accuracy = bandit.sub_bandits['random_forest'].optimizer['hpo'].incumbent_perf
    best_pred = rf.predict(test_data.data[0])
    test_accuracy = balanced_accuracy(test_data.data[1], best_pred)

    # es_pred = bandit._es_predict(test_data)
    # test_accuracy_with_ens = balanced_accuracy(test_data.data[1], es_pred)
    data = [dataset, validation_accuracy, test_accuracy, fe_exp_output, hpo_exp_output,
            _start_time]
    save_path = project_dir + '%s_%s_%s_%d_%d_%d_%d_%d.pkl' % (
        ausk_flag, opt_algo, dataset, trial_num, len(algorithms), seed, run_id, time_limit)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    del_path = './logs/'
    for i in os.listdir(del_path):
        file_data = del_path + "/" + i
        if os.path.isfile(file_data):
            os.remove(file_data)


def create_points(time_range, gap_num, x, y):
    gap_width = time_range / gap_num
    cur_idx = 1
    results = list()
    for i, plot_time in enumerate(x):
        while cur_idx <= gap_num:
            if cur_idx * gap_width > plot_time:
                break
            results.append(1 + y[i])
            cur_idx += 1
    while len(results) < gap_num:
        results.append(results[-1])
    return results


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = 100000
    modes = args.mode.split(',')
    rep = args.rep_num
    start_id = args.start_id
    time_limit = args.time_cost

    # Prepare random seeds.
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    algorithms = ['random_forest']

    dataset_list = dataset_str.split(',')

    for mode in modes:
        if mode != 'plot':
            for dataset in dataset_list:
                time_costs = list()

                for run_id in range(start_id, start_id + rep):
                    seed = int(seeds[run_id])
                    if mode == 'hmab':
                        evaluate_hmab(algorithms, dataset, run_id, trial_num, seed, time_limit=time_limit)
                    elif mode == 'ausk':
                        evaluate_autosklearn(algorithms, dataset, run_id, trial_num, seed, time_limit=time_limit)
                    else:
                        raise ValueError('Invalid parameter: %s' % mode)
        else:
            from matplotlib import pyplot as plt

            headers = ['dataset']
            method_ids = ['mfse_fixed', 'ausk_fixed']
            for mth in method_ids:
                headers.extend(['val-%s' % mth, 'test-%s' % mth])

            tbl_data = list()
            for dataset in dataset_list:
                row_data = [dataset]
                x = np.arange(100) * time_limit / 100
                y = dict()
                for mth in method_ids:
                    results = list()
                    y[mth] = list()
                    for run_id in range(rep):
                        seed = seeds[run_id]
                        time_t = time_limit
                        file_path = project_dir + '%s_%s_%d_%d_%d_%d_%d.pkl' % (
                            mth, dataset, trial_num, len(algorithms), seed, run_id, time_t)
                        if not os.path.exists(file_path):
                            print(file_path)
                            continue
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        if mth.startswith('hmab'):
                            val_acc, test_acc = data[1], data[2]
                        else:
                            val_acc, test_acc = data[1], data[2]
                        fe_output = data[3]
                        hpo_output = data[4]
                        start_time = data[5]
                        if 'mfse' in mth:
                            output = fe_output.copy()
                            output.update(hpo_output)
                            output = sorted(output.items(), key=lambda x: x[0])
                        else:
                            output = fe_output.copy()
                            output.update(hpo_output)
                            output = sorted(output.items(), key=lambda x: x[0])
                        plot_x, plot_y = list(), list()
                        best_val = float('inf')
                        for timestamp, data in output:
                            plot_x.append(timestamp - start_time)
                            try:
                                cur_val = min(data[2])
                            except:
                                cur_val = data[1]
                            if cur_val < best_val:
                                best_val = cur_val
                            plot_y.append(best_val)
                        plot_y = create_points(time_limit, 100, plot_x, plot_y)
                        y[mth].append(plot_y)
                        results.append([val_acc, test_acc])
                        # if mth.startswith('ausk'):
                        #     print('='*10)
                        #     print(val_acc, test_acc)
                        #     print(data[3])
                        #     print('='*10)

                    if len(results) == rep:
                        results = np.array(results)
                        stats_ = zip(np.mean(results, axis=0), np.std(results, axis=0))
                        string = ''
                        for mean_t, std_t in stats_:
                            string += u'%.3f\u00B1%.3f |' % (mean_t, std_t)
                        print(dataset, mth, '=' * 30)
                        print('%s-%s: mean\u00B1std' % (dataset, mth), string)
                        print('%s-%s: median' % (dataset, mth), np.median(results, axis=0))

                        for idx in range(results.shape[1]):
                            vals = results[:, idx]
                            median = np.median(vals)
                            if median == 0.:
                                row_data.append('-')
                            else:
                                row_data.append(u'%.4f' % median)
                    else:
                        row_data.extend(['-'] * 2)
                for l in y['ausk_fixed']:
                    pass
                    # print(len(l))
                plt.plot(x, np.mean(y['mfse_fixed'], axis=0), label='mfse')
                plt.plot(x, np.mean(y['ausk_fixed'], axis=0), label='ausk')
                plt.legend()
                if dataset == 'covertype':
                    plt.ylim(0.285, 0.32)
                elif dataset == 'higgs':
                    plt.ylim(0.2775, 0.29)
                elif dataset == 'codrna':
                    plt.ylim(0.022, 0.026)
                elif dataset == 'vehicle_sensIT':
                    plt.ylim(0.1275, 0.14)
                elif dataset == 'mnist_784':
                    plt.ylim(0.031, 0.04)
                plt.show()

                tbl_data.append(row_data)
            print(tabulate(tbl_data, headers, tablefmt='github'))
