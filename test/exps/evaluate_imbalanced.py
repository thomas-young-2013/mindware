import os
import sys
import time
import pickle
import argparse
import numpy as np
from tabulate import tabulate
from sklearn.metrics import make_scorer

sys.path.append(os.getcwd())

from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.utils.constants import CATEGORICAL
from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.components.metrics.cls_metrics import balanced_accuracy
from automlToolkit.components.utils.constants import MULTICLASS_CLS, BINARY_CLS

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--mode', type=str, choices=['hmab','plot'], default='plot')
parser.add_argument('--algo_num', type=int, default=15)
parser.add_argument('--time_cost', type=int, default=1200)
parser.add_argument('--trial_num', type=int, default=150)
parser.add_argument('--rep_num', type=int, default=10)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

project_dir = './'
per_run_time_limit = 180
opt_algo = 'rb_hpo'
hmab_flag = 'imb_hmab'


def evaluate_imbalanced(algorithms, dataset, run_id, trial_num, seed, time_limit=1200):
    print('%s-%s-%d: %d' % (hmab_flag, dataset, run_id, time_limit))
    _start_time = time.time()
    train_data, test_data = load_train_test_data(dataset)
    cls_task_type = BINARY_CLS if len(set(train_data.data[1])) == 2 else MULTICLASS_CLS
    # ACC or Balanced_ACC
    balanced_acc_metric = make_scorer(balanced_accuracy)
    bandit = FirstLayerBandit(cls_task_type, trial_num, algorithms, train_data,
                              output_dir='logs',
                              per_run_time_limit=per_run_time_limit,
                              dataset_name=dataset,
                              ensemble_size=50,
                              opt_algo=opt_algo,
                              metric=balanced_acc_metric,
                              fe_algo='bo',
                              seed=seed)
    bandit.optimize()
    model_desc = [bandit.nbest_algo_ids, bandit.optimal_algo_id, bandit.final_rewards, bandit.action_sequence]

    time_taken = time.time() - _start_time
    validation_accuracy = np.max(bandit.final_rewards)
    best_pred = bandit._best_predict(test_data)
    test_accuracy = balanced_accuracy(test_data.data[1], best_pred)
    es_pred = bandit._es_predict(test_data)
    test_accuracy_with_ens = balanced_accuracy(test_data.data[1], es_pred)
    data = [dataset, validation_accuracy, test_accuracy, test_accuracy_with_ens, time_taken, model_desc]
    print(model_desc)
    print(data[:4])

    save_path = project_dir + 'data/%s_%s_%s_%d_%d_%d_%d.pkl' % (
        hmab_flag, opt_algo, dataset, trial_num, len(algorithms), seed, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    modes = args.mode.split(',')
    rep = args.rep_num
    start_id = args.start_id

    # Prepare random seeds.
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    algorithms = ['balanced_bagging',
                  'balanced_random_forest',
                  'rusboost',
                  'easy_ensemble']

    dataset_list = dataset_str.split(',')

    for mode in modes:
        if mode != 'plot':
            for dataset in dataset_list:
                for run_id in range(start_id, start_id + rep):
                    seed = int(seeds[run_id])
                    if mode == 'hmab':
                        evaluate_imbalanced(algorithms, dataset, run_id, trial_num, seed)
                    else:
                        raise ValueError('Invalid parameter: %s' % mode)
        else:
            headers = ['dataset']
            method_ids = ['imb_hmab_rb_hpo']
            for mth in method_ids:
                headers.extend(['val-%s' % mth, 'test-%s' % mth])

            tbl_data = list()
            for dataset in dataset_list:
                row_data = [dataset]
                for mth in method_ids:
                    results = list()
                    for run_id in range(rep):
                        seed = seeds[run_id]
                        file_path = project_dir + 'data/%s_%s_%d_%d_%d_%d.pkl' % (
                            mth, dataset, trial_num, len(algorithms), seed, run_id)
                        if not os.path.exists(file_path):
                            continue
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        val_acc, test_acc = data[1], data[2]
                        results.append([val_acc, test_acc])

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

                tbl_data.append(row_data)
            print(tabulate(tbl_data, headers, tablefmt='github'))
