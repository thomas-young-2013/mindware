import os
import sys
import time
import pickle
import argparse
import tabulate
import numpy as np
import autosklearn.classification
from sklearn.metrics import balanced_accuracy_score

sys.path.append(os.getcwd())
from mindware.estimators import Classifier
from mindware.datasets.utils import load_train_test_data
from mindware.components.utils.constants import CATEGORICAL, MULTICLASS_CLS

parser = argparse.ArgumentParser()
dataset_set = 'dna,pollen,abalone,splice,madelon,spambase,wind,page-blocks(1),pc2,segment'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--methods', type=str, default='hmab,ausk')
parser.add_argument('--algo_num', type=int, default=15)
parser.add_argument('--time_limit', type=int, default=120)
parser.add_argument('--rep_num', type=int, default=10)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--ensemble', type=int, choices=[0, 1], default=1)
parser.add_argument('--eval_type', type=str, choices=['cv', 'holdout'], default='holdout')
parser.add_argument('--seed', type=int, default=1)

save_dir = './data/exp_results/exp2/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

args = parser.parse_args()
dataset_str = args.datasets
algo_num = args.algo_num
start_id = args.start_id
rep = args.rep_num
methods = args.methods.split(',')
time_limit = args.time_limit
eval_type = args.eval_type
enable_ensemble = bool(args.ensemble)
rep_num = args.rep_num

# Prepare random seeds.
np.random.seed(args.seed)
seeds = np.random.randint(low=1, high=10000, size=start_id + rep_num)

per_run_time_limit = 300
holdout_datasets = dataset_set.split(',')


def evaluate_hmab(algorithms, run_id,
                  time_limit=600,
                  dataset='credit',
                  eval_type='holdout',
                  enable_ens=True, seed=1):
    task_id = '[hmab][%s-%d-%d]' % (dataset, len(algorithms), time_limit)
    _start_time = time.time()
    train_data, test_data = load_train_test_data(dataset, task_type=MULTICLASS_CLS)
    if enable_ens is True:
        ensemble_method = 'ensemble_selection'
    else:
        ensemble_method = None

    clf = Classifier(time_limit=time_limit,
                     amount_of_resource=None,
                     output_dir=save_dir,
                     ensemble_method=ensemble_method,
                     evaluation=eval_type,
                     metric='bal_acc',
                     n_jobs=1)
    clf.fit(train_data)
    clf.refit()
    pred = clf.predict(test_data)
    test_score = balanced_accuracy_score(test_data.data[1], pred)
    timestamps, perfs = clf.get_val_stats()
    validation_score = np.max(perfs)
    print('Dataset          : %s' % dataset)
    print('Validation/Test score : %f - %f' % (validation_score, test_score))

    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        stats = [timestamps, perfs]
        pickle.dump([validation_score, test_score, stats], f)


def evaluate_autosklearn(algorithms, rep_id,
                         dataset='credit', time_limit=1200, seed=1,
                         enable_ens=True, enable_meta_learning=True,
                         eval_type='holdout'):
    print('%s\nDataset: %s, Run_id: %d, Budget: %d.\n%s' % ('=' * 50, dataset, rep_id, time_limit, '=' * 50))
    task_id = '[ausk%d][%s-%d-%d]' % (enable_ens, dataset, len(algorithms), time_limit)
    if enable_ens:
        ensemble_size, ensemble_nbest = 50, 50
    else:
        ensemble_size, ensemble_nbest = 1, 1
    if enable_meta_learning:
        init_config_via_metalearning = 25
    else:
        init_config_via_metalearning = 0

    include_models = None

    if eval_type == 'holdout':
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=int(time_limit),
            per_run_time_limit=per_run_time_limit,
            n_jobs=1,
            include_estimators=include_models,
            ensemble_memory_limit=16384,
            ml_memory_limit=16384,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            initial_configurations_via_metalearning=init_config_via_metalearning,
            seed=int(seed),
            resampling_strategy='holdout',
            resampling_strategy_arguments={'train_size': 0.67}
        )
    else:
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=int(time_limit),
            per_run_time_limit=per_run_time_limit,
            n_jobs=1,
            include_estimators=include_models,
            ensemble_memory_limit=16384,
            ml_memory_limit=16384,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            initial_configurations_via_metalearning=init_config_via_metalearning,
            seed=seed,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5}
        )

    print(automl)
    raw_data, test_raw_data = load_train_test_data(dataset, task_type=MULTICLASS_CLS)
    X, y = raw_data.data
    X_test, y_test = test_raw_data.data
    feat_type = ['Categorical' if _type == CATEGORICAL else 'Numerical'
                 for _type in raw_data.feature_types]
    from autosklearn.metrics import balanced_accuracy as balanced_acc
    automl.fit(X.copy(), y.copy(), feat_type=feat_type, metric=balanced_acc)
    model_desc = automl.show_models()
    str_stats = automl.sprint_statistics()
    valid_results = automl.cv_results_['mean_test_score']
    time_records = automl.cv_results_['mean_fit_time']
    plot_x = convert_ausk_to_plot(time_records, time_limit)

    validation_score = np.max(valid_results)

    # Test performance.
    automl.refit(X.copy(), y.copy())
    predictions = automl.predict(X_test)
    test_score = balanced_accuracy_score(y_test, predictions)

    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(str_stats)
    print(model_desc)
    print('Validation Accuracy:', validation_score)
    print("Test Accuracy      :", test_score)

    save_path = save_dir + '%s-%d.pkl' % (task_id, rep_id)
    with open(save_path, 'wb') as f:
        stats = [plot_x, valid_results]
        pickle.dump([validation_score, test_score, stats], f)


def convert_ausk_to_plot(time_array, total_cost):
    total_fit_time = sum(time_array)
    per_other_time = (total_cost - total_fit_time) / (len(time_array) - 1)
    convert_x = list()
    prev_t = 0
    for i, t in enumerate(time_array):
        if i == 0:
            cur_t = t
            prev_t = cur_t
        else:
            cur_t = t + prev_t + per_other_time
            prev_t = cur_t
        convert_x.append(cur_t)
    return convert_x


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1, task_type=MULTICLASS_CLS)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == "__main__":
    algorithms = ['adaboost', 'random_forest',
                  'libsvm_svc', 'sgd',
                  'extra_trees', 'decision_tree',
                  'liblinear_svc', 'k_nearest_neighbors',
                  'passive_aggressive', 'gradient_boosting',
                  'lda', 'qda',
                  'multinomial_nb', 'gaussian_nb', 'bernoulli_nb'
                  ]

    dataset_list = dataset_str.split(',')
    check_datasets(dataset_list)

    if time_limit == 0:
        time_limits = [150, 300, 600, 1200, 3600]
    else:
        time_limits = [time_limit]

    for time_limit in time_limits:
        for dataset in dataset_list:
            for mth in methods:
                if mth == 'plot':
                    break

                for run_id in range(start_id, start_id + rep):
                    seed = int(seeds[run_id])
                    if mth == 'hmab':
                        evaluate_hmab(algorithms, run_id, dataset=dataset, seed=seed,
                                      eval_type=eval_type,
                                      time_limit=time_limit,
                                      enable_ens=enable_ensemble)
                    elif mth == 'ausk':
                        evaluate_autosklearn(algorithms, run_id,
                                             dataset=dataset, time_limit=time_limit, seed=seed,
                                             enable_ens=enable_ensemble,
                                             eval_type=eval_type)
                    else:
                        raise ValueError('Invalid method name: %s.' % mth)

    if methods[-1] == 'plot':
        time_limits = [150, 300, 600, 1200, 2400]
        headers = ['dataset']
        method_ids = ['hmab', 'ausk%d' % enable_ensemble]
        for time_limit in time_limits:
            headers.extend(['%d' % time_limit])

        tbl_data, figure_data = list(), list()
        for dataset in dataset_list:
            for mth in method_ids:
                row_data, plot_data = ['%s-%s' % (dataset, mth)], list()
                for time_limit in time_limits:
                    results = list()
                    for run_id in range(rep):
                        task_id = '[%s][%s-%d-%d]' % (mth, dataset, len(algorithms), time_limit)
                        file_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
                        if not os.path.exists(file_path):
                            continue
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        val_acc, test_acc, _ = data
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
                        median = np.median(results[:, 1])
                        row_data.append(u'%.4f' % median)
                        plot_data.append(median)
                else:
                    row_data.extend(['-'] * len(time_limits))
                tbl_data.append(row_data)
                figure_data.append(plot_data)

        print(tabulate.tabulate(tbl_data, headers, tablefmt='github'))
