import os
import sys
import time
import pickle
import argparse
import tabulate
import numpy as np
import autosklearn.classification
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.datasets.utils import load_data, load_train_test_data
from automlToolkit.components.ensemble.ensemble_builder import EnsembleBuilder
from automlToolkit.components.utils.constants import CATEGORICAL

parser = argparse.ArgumentParser()
dataset_set = 'yeast,vehicle,diabetes,spectf,credit,' \
              'ionosphere,lymphography,messidor_features,winequality_red,fri_c1,quake,satimage'
parser.add_argument('--eval_type', type=str, choices=['cv', 'holdout'], default='holdout')
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--methods', type=str, default='hmab,ausk')
parser.add_argument('--algo_num', type=int, default=15)
parser.add_argument('--trial_num', type=int, default=100)
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--n', type=int, default=4)
parser.add_argument('--meta', type=int, default=0)
parser.add_argument('--time_costs', type=str, default='1200')
parser.add_argument('--seed', type=int, default=1)

save_dir = './data/exp_results/exp2/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

per_run_time_limit = 240


def evaluate_hmab(algorithms, run_id, dataset='credit', trial_num=200,
                  n_jobs=1, meta_configs=0, seed=1, eval_type='holdout'):
    task_id = '%s-hmab-%d-%d' % (dataset, len(algorithms), trial_num)
    _start_time = time.time()
    raw_data, test_raw_data = load_train_test_data(dataset, random_state=seed)
    bandit = FirstLayerBandit(trial_num, algorithms, raw_data,
                              output_dir='logs/%s/' % task_id,
                              per_run_time_limit=per_run_time_limit,
                              dataset_name='%s-%d' % (dataset, run_id),
                              n_jobs=n_jobs,
                              meta_configs=meta_configs,
                              seed=seed,
                              eval_type=eval_type)
    bandit.optimize()
    time_cost = int(time.time() - _start_time)
    print(bandit.final_rewards)
    print(bandit.action_sequence)

    validation_accuracy = np.max(bandit.final_rewards)
    validation_accuracy_without_ens = bandit.validate()
    assert np.isclose(validation_accuracy, validation_accuracy_without_ens)
    test_accuracy_with_ens = EnsembleBuilder(bandit).score(test_raw_data)

    print('Dataset                     : %s' % dataset)
    print('Validation score without ens: %f' % validation_accuracy)
    print("Test score with ensemble    : %f" % test_accuracy_with_ens)

    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        stats = [time_cost]
        pickle.dump([validation_accuracy, test_accuracy_with_ens, stats], f)
    del bandit
    return time_cost


def load_hmab_time_costs(start_id, rep, dataset, n_algo, trial_num):
    task_id = '%s-hmab-%d-%d' % (dataset, n_algo, trial_num)
    time_costs = list()
    for run_id in range(start_id, start_id + rep):
        save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
        with open(save_path, 'rb') as f:
            time_cost = pickle.load(f)[2][0]
            time_costs.append(time_cost)
    assert len(time_costs) == rep
    return time_costs


def evaluate_autosklearn(algorithms, rep_id, trial_num=100,
                         dataset='credit', time_limit=1200, seed=1,
                         enable_ens=True, enable_meta_learning=False,
                         eval_type='holdout', n_jobs=1):
    print('%s\nDataset: %s, Run_id: %d, Budget: %d.\n%s' % ('=' * 50, dataset, rep_id, time_limit, '=' * 50))
    task_id = '%s-%s-%d-%d' % (dataset, 'ausk-full', len(algorithms), trial_num)
    if enable_ens:
        ensemble_size, ensemble_nbest = 50, 50
    else:
        ensemble_size, ensemble_nbest = 1, 1
    if enable_meta_learning:
        init_config_via_metalearning = 25
    else:
        init_config_via_metalearning = 0

    include_models = algorithms

    if eval_type == 'holdout':
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=int(time_limit),
            per_run_time_limit=per_run_time_limit,
            n_jobs=n_jobs,
            include_estimators=include_models,
            ensemble_memory_limit=12288,
            ml_memory_limit=12288,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            initial_configurations_via_metalearning=init_config_via_metalearning,
            seed=seed,
            resampling_strategy='holdout',
            resampling_strategy_arguments={'train_size': 0.8}
        )
    else:
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=int(time_limit),
            per_run_time_limit=per_run_time_limit,
            n_jobs=n_jobs,
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
    raw_data, test_raw_data = load_train_test_data(dataset, random_state=seed)
    X, y = raw_data.data
    X_test, y_test = test_raw_data.data
    feat_type = ['Categorical' if _type == CATEGORICAL else 'Numerical'
                 for _type in raw_data.feature_types]
    automl.fit(X.copy(), y.copy(), feat_type=feat_type)
    model_desc = automl.show_models()
    str_stats = automl.sprint_statistics()
    valid_results = automl.cv_results_['mean_test_score']
    time_records = automl.cv_results_['mean_fit_time']
    validation_accuracy = np.max(valid_results)

    # Test performance.
    if eval_type == 'cv':
        automl.refit(X.copy(), y.copy())
    predictions = automl.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)

    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(str_stats)
    print(model_desc)
    print('Validation Accuracy:', validation_accuracy)
    print("Test Accuracy      :", test_accuracy)

    save_path = save_dir + '%s-%d.pkl' % (task_id, rep_id)
    with open(save_path, 'wb') as f:
        stats = [model_desc, str_stats, valid_results, time_records, time_limit]
        pickle.dump([validation_accuracy, test_accuracy, stats], f)


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    start_id = args.start_id
    rep = args.rep_num
    n_jobs = args.n
    meta_configs = args.meta
    methods = args.methods.split(',')
    time_costs = [int(item) for item in args.time_costs.split(',')]
    eval_type = args.eval_type

    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    if algo_num == 4:
        algorithms = ['extra_trees', 'sgd', 'decision_tree', 'passive_aggressive']
    elif algo_num == 8:
        algorithms = ['passive_aggressive', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                      'adaboost', 'random_forest', 'extra_trees', 'decision_tree']
    elif algo_num == 15:
        algorithms = ['adaboost', 'random_forest',
                      'libsvm_svc', 'sgd',
                      'extra_trees', 'decision_tree',
                      'k_nearest_neighbors', 'liblinear_svc',
                      'passive_aggressive', 'xgradient_boosting',
                      'lda', 'qda',
                      'multinomial_nb', 'gaussian_nb', 'bernoulli_nb']
    else:
        raise ValueError('Invalid algo num: %d' % algo_num)

    dataset_list = dataset_str.split(',')
    check_datasets(dataset_list)

    for dataset in dataset_list:
        for mth in methods:
            if mth == 'plot':
                break

            if mth.startswith('ausk'):
                time_costs = load_hmab_time_costs(start_id, rep, dataset, len(algorithms), trial_num)
                print(time_costs)
                median = np.median(time_costs)
                time_costs = [median] * rep
                print(median, time_costs)

            for run_id in range(start_id, start_id + rep):
                seed = int(seeds[run_id])
                if mth == 'hmab':
                    time_cost = evaluate_hmab(algorithms, run_id, dataset,
                                              trial_num=trial_num, seed=seed,
                                              n_jobs=n_jobs, meta_configs=meta_configs,
                                              eval_type=eval_type)
                elif mth == 'ausk':
                    time_cost = time_costs[run_id - start_id]
                    evaluate_autosklearn(algorithms, run_id, trial_num, dataset,
                                         time_cost, seed=seed,
                                         enable_meta_learning=True, enable_ens=True,
                                         n_jobs=n_jobs, eval_type=eval_type)
                else:
                    raise ValueError('Invalid method name: %s.' % mth)

    if methods[-1] == 'plot':
        headers = ['dataset']
        method_ids = ['hmab', 'ausk-full']
        for mth in method_ids:
            headers.extend(['val-%s' % mth, 'test-%s' % mth])

        tbl_data = list()
        for dataset in dataset_list:
            row_data = [dataset]
            for mth in method_ids:
                results = list()
                for run_id in range(rep):
                    task_id = '%s-%s-%d-%d' % (dataset, mth, len(algorithms), trial_num)
                    file_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    val_acc, test_acc, _tmp = data
                    results.append([val_acc, test_acc])
                if len(results) == rep:
                    results = np.array(results)
                    print('%s-%s' % (dataset, mth), '=' * 20)
                    stats_ = zip(np.mean(results, axis=0), np.std(results, axis=0))
                    string = ''
                    for mean_t, std_t in stats_:
                        string += u'%.3f\u00B1%.3f |' % (mean_t, std_t)
                    print(string)
                    print('%s-%s' % (dataset, mth), '=' * 20)
                    for idx in range(results.shape[1]):
                        vals = results[:, idx]
                        mean_, std_ = np.mean(vals), np.std(vals)
                        if mean_ == 0.:
                            row_data.append('-')
                        else:
                            row_data.append(u'%.3f\u00B1%.3f' % (mean_, std_))
                else:
                    row_data.extend(['-'] * 2)

            tbl_data.append(row_data)
        print(tabulate.tabulate(tbl_data, headers, tablefmt='github'))
