import os
import sys
import time
import pickle
import argparse
import numpy as np
import autosklearn.classification
from tabulate import tabulate

sys.path.append(os.getcwd())

from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.utils.constants import CATEGORICAL
from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.components.metrics.cls_metrics import balanced_accuracy
from automlToolkit.components.ensemble.ensemble_builder import EnsembleBuilder

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--mode', type=str, choices=['ausk', 'hmab', 'plot'], default='plot')
parser.add_argument('--algo_num', type=int, default=15)
parser.add_argument('--time_cost', type=int, default=1200)
parser.add_argument('--trial_num', type=int, default=150)
parser.add_argument('--rep_num', type=int, default=10)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

project_dir = './'
per_run_time_limit = 150
opt_algo = 'alter_hpo'


def evaluate_1stlayer_bandit(algorithms, dataset, run_id, trial_num, seed, time_limit=1200):
    _start_time = time.time()
    train_data, test_data = load_train_test_data(dataset)
    bandit = FirstLayerBandit(trial_num, algorithms, train_data,
                              output_dir='logs',
                              per_run_time_limit=per_run_time_limit,
                              dataset_name=dataset,
                              opt_algo=opt_algo,
                              seed=seed)
    bandit.optimize()
    model_desc = [bandit.nbest_algo_ids, bandit.optimal_algo_id, bandit.final_rewards, bandit.action_sequence]

    time_taken = time.time() - _start_time
    validation_accuracy = np.max(bandit.final_rewards)
    test_accuracy = bandit.score(test_data, metric_func=balanced_accuracy)
    test_accuracy_with_ens = EnsembleBuilder(bandit).score(test_data, metric_func=balanced_accuracy)
    data = [dataset, validation_accuracy, test_accuracy, test_accuracy_with_ens, time_taken, model_desc]
    print(model_desc)

    print(data[:4])

    save_path = project_dir + 'data/hmab_%s_%s_%d_%d_%d_%d.pkl' % (
        opt_algo, dataset, trial_num, len(algorithms), seed, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def load_hmab_time_costs(start_id, rep, dataset, n_algo, trial_num, seeds):
    time_costs = list()
    for run_id in range(start_id, start_id + rep):
        seed = seeds[run_id]
        save_path = project_dir + 'data/hmab_%s_%s_%d_%d_%d_%d.pkl' % (opt_algo, dataset, trial_num,
                                                                       n_algo, seed, run_id)
        with open(save_path, 'rb') as f:
            time_cost_ = int(pickle.load(f)[4])
            time_costs.append(time_cost_)
    assert len(time_costs) == rep
    print(time_costs)
    return time_costs


def evaluate_autosklearn(algorithms, dataset, run_id, trial_num, seed, time_limit=1200):
    print('==> Start to evaluate', dataset, 'budget', time_limit)
    include_models = algorithms
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_limit,
        per_run_time_limit=per_run_time_limit,
        include_preprocessors=None,
        exclude_preprocessors=None,
        n_jobs=1,
        include_estimators=include_models,
        ensemble_memory_limit=8192,
        ml_memory_limit=8192,
        ensemble_size=1,
        ensemble_nbest=1,
        initial_configurations_via_metalearning=0,
        seed=int(seed),
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67}
    )
    print(automl)

    train_data, test_data = load_train_test_data(dataset)
    X, y = train_data.data
    feat_type = ['Categorical' if _type == CATEGORICAL else 'Numerical'
                 for _type in train_data.feature_types]

    from autosklearn.metrics import balanced_accuracy
    automl.fit(X.copy(), y.copy(), metric=balanced_accuracy, feat_type=feat_type)
    model_desc = automl.show_models()
    print(model_desc)
    val_result = np.max(automl.cv_results_['mean_test_score'])
    print('Best validation accuracy', val_result)

    X_test, y_test = test_data.data
    automl.refit(X.copy(), y.copy())
    y_pred = automl.predict(X_test)
    test_result = balanced_accuracy(y_test, y_pred)
    print('Test accuracy', test_result)
    save_path = project_dir + 'data/ausk_vanilla_%s_%d_%d_%d_%d.pkl' % (
        dataset, trial_num, len(algorithms), seed, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, val_result, test_result, model_desc], f)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    mode = args.mode
    rep = args.rep_num
    start_id = args.start_id

    # Prepare random seeds.
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    algorithms = ['adaboost', 'random_forest',
                  'libsvm_svc', 'sgd',
                  'extra_trees', 'decision_tree',
                  'liblinear_svc', 'k_nearest_neighbors',
                  'passive_aggressive', 'xgradient_boosting',
                  'lda', 'qda',
                  'multinomial_nb', 'gaussian_nb', 'bernoulli_nb'
                  ]
    # algorithms = ['adaboost', 'random_forest',
    #               'extra_trees'
    #               ]

    dataset_list = dataset_str.split(',')

    if mode != 'plot':
        for dataset in dataset_list:
            time_costs = list()
            if mode == 'ausk':
                time_costs = load_hmab_time_costs(start_id, rep, dataset, len(algorithms), trial_num, seeds)

            for run_id in range(start_id, start_id + rep):
                seed = int(seeds[run_id])
                if mode == 'hmab':
                    evaluate_1stlayer_bandit(algorithms, dataset, run_id, trial_num, seed)
                elif mode == 'ausk':
                    time_taken = time_costs[run_id-start_id]
                    evaluate_autosklearn(algorithms, dataset, run_id, trial_num, seed, time_limit=time_taken)
                else:
                    raise ValueError('Invalid parameter: %s' % mode)
    else:
        headers = ['dataset']
        method_ids = ['hmab_alter_hpo', 'ausk_vanilla']
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

            tbl_data.append(row_data)
        print(tabulate(tbl_data, headers, tablefmt='github'))
