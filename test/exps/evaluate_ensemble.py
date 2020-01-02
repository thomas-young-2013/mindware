import os
import sys
import time
import pickle
import argparse
import numpy as np
import autosklearn.classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

from automlToolkit.components.evaluator import get_estimator
from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.datasets.utils import load_data, load_train_test_data
from automlToolkit.components.ensemble.ensemble_selection import EnsembleSelection
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.utils.constants import CATEGORICAL

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--mode', type=str, choices=['ausk', 'mab', 'benchmark'], default='benchmark')
parser.add_argument('--algo_num', type=int, default=8)
parser.add_argument('--time_cost', type=int, default=1200)
parser.add_argument('--trial_num', type=int, default=100)
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

save_dir = './data/ens_result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

per_run_time_limit = 150


def evaluate_1stlayer_bandit(algorithms, run_id, dataset='credit', trial_num=200, seed=1):
    task_id = '%s-hmab-%d-%d' % (dataset, len(algorithms), trial_num)
    _start_time = time.time()
    raw_data, test_raw_data = load_train_test_data(dataset)
    bandit = FirstLayerBandit(trial_num, algorithms, raw_data,
                              output_dir='logs/%s/' % task_id,
                              per_run_time_limit=per_run_time_limit,
                              dataset_name='%s-%d' % (dataset, run_id),
                              seed=seed,
                              eval_type='holdout')
    bandit.optimize()
    time_cost = time.time() - _start_time
    print(bandit.final_rewards)
    print(bandit.action_sequence)

    validation_accuracy = np.max(bandit.final_rewards)

    validation_accuracy_without_ens = bandit.validate()
    test_accuracy_without_ens = bandit.score(test_raw_data)
    test_accuracy = ensemble_implementation_examples(bandit, test_raw_data)

    print('Validation score without ens', validation_accuracy_without_ens, validation_accuracy)
    print("Test score without ensemble: %s - %f" % (dataset, test_accuracy_without_ens))
    print("Test score With ensemble: %s - %f" % (dataset, test_accuracy))

    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        stats = [time_cost]
        pickle.dump([validation_accuracy, test_accuracy, stats], f)
    return time_cost


def ensemble_implementation_examples(bandit: FirstLayerBandit, test_data: DataNode):
    from sklearn.model_selection import train_test_split
    from autosklearn.metrics import accuracy
    from sklearn.metrics import accuracy_score
    n_best = 20
    stats = bandit.fetch_ensemble_members(test_data)
    seed = stats['split_seed']
    train_predictions = []
    test_predictions = []
    for algo_id in bandit.arms:
        X, y = stats[algo_id]['train_dataset'].data
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        X_test, y_test = stats[algo_id]['test_dataset'].data
        configs = stats[algo_id]['configurations']
        performance = stats[algo_id]['performance']
        best_index = np.argsort(-np.array(performance))
        best_configs = [configs[i] for i in best_index[:n_best]]

        for config in best_configs:
            # Build the ML estimator.
            _, estimator = get_estimator(config)
            # print(X_train.shape, X_test.shape)
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict_proba(X_valid)
            train_predictions.append(y_pred)
            y_pred = estimator.predict_proba(X_test)
            test_predictions.append(y_pred)

    es = EnsembleSelection(ensemble_size=50, task_type=1, metric=accuracy, random_state=np.random.RandomState(42))
    es.fit(train_predictions, y_valid, identifiers=None)
    y_pred = es.predict(test_predictions)
    y_pred = np.argmax(y_pred, axis=-1)
    test_score = accuracy_score(y_test, y_pred)
    return test_score


def evaluate_autosklearn(algorithms, rep_id, dataset='credit', time_limit=1200, seed=1):
    print('%s\nDataset: %s, Run_id: %d, Budget: %d.\n%s' % ('='*50, dataset, rep_id, time_limit, '='*50))

    include_models = algorithms
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_limit,
        per_run_time_limit=per_run_time_limit,
        n_jobs=1,
        include_estimators=include_models,
        ensemble_memory_limit=8192,
        ml_memory_limit=8192,
        ensemble_size=50,
        initial_configurations_via_metalearning=0,
        seed=seed,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.8}
    )
    print(automl)
    raw_data, test_raw_data = load_train_test_data(dataset)
    X, y = raw_data.data
    X_test, y_test = test_raw_data.data
    feat_type = ['Categorical' if _type == CATEGORICAL else 'Numerical'
                 for _type in raw_data.feature_types]
    automl.fit(X.copy(), y.copy(), feat_type=feat_type)
    model_desc = automl.show_models()
    test_results = automl.cv_results_['mean_test_score']
    time_records = automl.cv_results_['mean_fit_time']
    validation_accuracy = np.max(test_results)
    predictions = automl.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print(model_desc)
    print('Validation Accuracy', validation_accuracy)
    print("Test Accuracy", test_accuracy)

    save_path = save_dir + 'ausk_%s_%d_%d.pkl' % (dataset, len(algorithms), rep_id)
    with open(save_path, 'wb') as f:
        stats = [model_desc, test_results, time_records, time_limit]
        pickle.dump([validation_accuracy, test_accuracy, stats], f)


def benchmark(algorithms, run_id, dataset, trial_num, seed):
    time_cost = evaluate_1stlayer_bandit(algorithms, run_id, dataset, trial_num=trial_num, seed=seed)
    evaluate_autosklearn(algorithms, run_id, dataset, int(time_cost), seed=seed)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    start_id = args.start_id
    rep = args.rep_num
    mode = args.mode
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    algorithms = ['k_nearest_neighbors', 'libsvm_svc', 'random_forest', 'adaboost']
    if algo_num == 8:
        algorithms = ['lda', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                      'adaboost', 'random_forest', 'extra_trees', 'decision_tree']

    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set
    else:
        dataset_list = dataset_str.split(',')

    for dataset in dataset_list:
        for run_id in range(start_id, start_id+rep):
            seed = int(seeds[run_id])
            time_cost = evaluate_1stlayer_bandit(algorithms, run_id, dataset, trial_num=trial_num, seed=seed)
            # benchmark(algorithms, run_id, dataset, trial_num, seed=seeds[run_id])
            # evaluate_autosklearn(algorithms, run_id, dataset, 120, seed=seed)
