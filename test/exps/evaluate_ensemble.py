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

from automlToolkit.components.evaluators.evaluator import get_estimator
from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.ensemble.ensemble_selection import EnsembleSelection
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.ensemble.ensemble_builder import EnsembleBuilder
from automlToolkit.components.utils.constants import CATEGORICAL

parser = argparse.ArgumentParser()
dataset_set = 'yeast,vehicle,diabetes,spectf,credit,' \
              'ionosphere,lymphography,messidor_features,winequality_red,fri_c1,quake,satimage'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--methods', type=str, default='hmab,ausk-full')
parser.add_argument('--algo_num', type=int, default=8)
parser.add_argument('--trial_num', type=int, default=100)
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--meta', type=int, default=0)
parser.add_argument('--time_costs', type=str, default='1200')
parser.add_argument('--seed', type=int, default=1)

save_dir = './data/ensemble_evaluation/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

per_run_time_limit = 240


def evaluate_1stlayer_bandit(algorithms, run_id, dataset='credit', trial_num=200, n_jobs=1, meta_configs=0, seed=1):
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
                              eval_type='holdout')
    bandit.optimize()
    time_cost = int(time.time() - _start_time)
    print(bandit.final_rewards)
    print(bandit.action_sequence)

    validation_accuracy_without_ens0 = np.max(bandit.final_rewards)
    validation_accuracy_without_ens1 = bandit.validate()
    assert np.isclose(validation_accuracy_without_ens0, validation_accuracy_without_ens1)

    test_accuracy_without_ens = bandit.score(test_raw_data)
    # For debug.
    mode = True
    if mode:
        test_accuracy_with_ens0 = ensemble_implementation_examples(bandit, test_raw_data)
        test_accuracy_with_ens1 = EnsembleBuilder(bandit).score(test_raw_data)

        print('Dataset                     : %s' % dataset)
        print('Validation score without ens: %f - %f' % (
            validation_accuracy_without_ens0, validation_accuracy_without_ens1))
        print("Test score without ensemble : %f" % test_accuracy_without_ens)
        print("Test score with ensemble    : %f - %f" % (test_accuracy_with_ens0, test_accuracy_with_ens1))

        save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
        with open(save_path, 'wb') as f:
            stats = [time_cost, test_accuracy_with_ens0, test_accuracy_with_ens1, test_accuracy_without_ens]
            pickle.dump([validation_accuracy_without_ens0, test_accuracy_with_ens1, stats], f)
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


def ensemble_implementation_examples(bandit: FirstLayerBandit, test_data: DataNode):
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import accuracy_score
    from autosklearn.metrics import accuracy
    n_best = 20
    stats = bandit.fetch_ensemble_members(test_data)
    seed = stats['split_seed']
    test_size = 0.2
    train_predictions = []
    test_predictions = []
    for algo_id in bandit.nbest_algo_ids:
        X, y = stats[algo_id]['train_dataset'].data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
        for train_index, test_index in sss.split(X, y):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]

        X_test, y_test = stats[algo_id]['test_dataset'].data
        configs = stats[algo_id]['configurations']
        performance = stats[algo_id]['performance']
        best_index = np.argsort(-np.array(performance))
        best_configs = [configs[i] for i in best_index[:n_best]]

        for config in best_configs:
            try:
                # Build the ML estimator.
                _, estimator = get_estimator(config)
                # print(X_train.shape, X_test.shape)
                estimator.fit(X_train, y_train)
                y_valid_pred = estimator.predict_proba(X_valid)
                y_test_pred = estimator.predict_proba(X_test)
                train_predictions.append(y_valid_pred)
                test_predictions.append(y_test_pred)
            except Exception as e:
                print(str(e))

    es = EnsembleSelection(ensemble_size=50, task_type=1,
                           metric=accuracy, random_state=np.random.RandomState(seed))
    assert len(train_predictions) == len(test_predictions)
    es.fit(train_predictions, y_valid, identifiers=None)
    y_pred = es.predict(test_predictions)
    y_pred = np.argmax(y_pred, axis=-1)
    test_score = accuracy_score(y_test, y_pred)
    return test_score


def evaluate_autosklearn(algorithms, rep_id, trial_num=100,
                         dataset='credit', time_limit=1200, seed=1,
                         enable_ens=True, enable_meta_learning=False):
    print('%s\nDataset: %s, Run_id: %d, Budget: %d.\n%s' % ('=' * 50, dataset, rep_id, time_limit, '=' * 50))
    ausk_id = 'ausk' if enable_ens else 'ausk-no-ens'
    ausk_id += '-meta' if enable_meta_learning else ''
    task_id = '%s-%s-%d-%d' % (dataset, ausk_id, len(algorithms), trial_num)
    if enable_ens:
        ensemble_size, ensemble_nbest = 50, 50
    else:
        ensemble_size, ensemble_nbest = 1, 1
    if enable_meta_learning:
        init_config_via_metalearning = 25
    else:
        init_config_via_metalearning = 0

    include_models = algorithms

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=int(time_limit),
        per_run_time_limit=per_run_time_limit,
        n_jobs=1,
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
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)
    """ Algorithm Set:
    algorithms = ['liblinear_svc', 'k_nearest_neighbors', 
                  'libsvm_svc', 'sgd',
                  'adaboost', 'random_forest', 
                  'extra_trees', 'decision_tree', 
                  'passive_aggressive', 
                  'lda', 'qda',
                  'multinomial_nb', 'gaussian_nb']
    
    """

    # algorithms = ['k_nearest_neighbors', 'libsvm_svc', 'random_forest', 'adaboost']
    algorithms = ['extra_trees', 'sgd',
                  'decision_tree', 'passive_aggressive']
    if algo_num == 8:
        algorithms = ['passive_aggressive', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                      'adaboost', 'random_forest', 'extra_trees', 'decision_tree']

    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set.split(',')
    else:
        dataset_list = dataset_str.split(',')

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
                    time_cost = evaluate_1stlayer_bandit(algorithms, run_id,
                                                         dataset, trial_num=trial_num, seed=seed, n_jobs=n_jobs,
                                                         meta_configs=meta_configs)
                elif mth == 'ausk':
                    time_cost = time_costs[run_id - start_id]
                    evaluate_autosklearn(algorithms, run_id, trial_num,
                                         dataset, time_cost, seed=seed)
                elif mth == 'ausk-no-ens':
                    time_cost = time_costs[run_id - start_id]
                    evaluate_autosklearn(algorithms, run_id, trial_num, dataset,
                                         time_cost, seed=seed, enable_ens=False)
                elif mth == 'ausk-full':
                    time_cost = time_costs[run_id - start_id]
                    evaluate_autosklearn(algorithms, run_id, trial_num, dataset,
                                         time_cost, seed=seed, enable_meta_learning=True)
                else:
                    raise ValueError('Invalid method name: %s.' % mth)

    if methods[-1] == 'plot':
        headers = ['dataset']
        method_ids = ['hmab', 'ausk', 'ausk-no-ens', 'ausk-meta']
        for mth in method_ids:
            headers.extend(['val-%s' % mth, 't1-%s' % mth, 't2-%s' % mth])

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
                    val_acc, test_acc_ens, _tmp = data
                    if mth == 'hmab':
                        test_acc_no_ens = _tmp[3]
                        test_acc_ens = np.max([_tmp[1], _tmp[2], test_acc_ens])
                    elif mth == 'ausk':
                        test_acc_no_ens = 0.
                    elif mth == 'ausk-no-ens':
                        test_acc_no_ens = test_acc_ens
                        test_acc_ens = 0
                    elif mth == 'ausk-meta':
                        test_acc_no_ens = 0
                        test_acc_ens = test_acc_ens
                    else:
                        raise ValueError('invalid method: %s' % mth)

                    results.append([val_acc, test_acc_no_ens, test_acc_ens])
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
                    row_data.extend(['-'] * 3)

            tbl_data.append(row_data)
        print(tabulate.tabulate(tbl_data, headers, tablefmt='github'))
