import os
import sys
import pickle
import argparse
import numpy as np
import autosklearn.classification
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from solnml.datasets.utils import load_train_test_data
from solnml.components.utils.constants import CATEGORICAL

parser = argparse.ArgumentParser()
dataset_set = 'yeast,vehicle,diabetes,spectf,credit,' \
              'ionosphere,lymphography,messidor_features,winequality_red,fri_c1,quake,satimage'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--algo_num', type=int, default=8)
parser.add_argument('--trial_num', type=int, default=100)
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--time_cost', type=int, default=-1)
parser.add_argument('--seed', type=int, default=1)

save_dir = './data/ens_result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

per_run_time_limit = 150


def load_hmab_time_costs(start_id, rep, dataset, n_algo, trial_num):
    task_id = '%s-hmab-%d-%d' % (dataset, n_algo, trial_num)
    time_costs = list()
    for run_id in range(start_id, start_id+rep):
        save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
        with open(save_path, 'rb') as f:
            time_cost = pickle.load(f)[2][0]
            time_costs.append(time_cost)
    assert len(time_costs) == rep
    return time_costs


def evaluate_autosklearn(algorithms, rep_id, trial_num=100, dataset='credit',
                         time_limit=1200, seed=1, ensemble_enable=True):
    print('%s\nDataset: %s, Run_id: %d, Budget: %d.\n%s' % ('='*50, dataset, rep_id, time_limit, '='*50))
    mth_id = 'ausk-ens' if ensemble_enable else 'ausk'
    task_id = '%s-%s-%d-%d' % (dataset, mth_id, len(algorithms), trial_num)
    include_models = algorithms
    if ensemble_enable:
        ensemble_size = 50
        ensem_nbest = len(algorithms)*20
    else:
        ensemble_size = 1
        ensem_nbest = 1

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=int(time_limit),
        per_run_time_limit=per_run_time_limit,
        n_jobs=1,
        include_estimators=include_models,
        ensemble_memory_limit=12288,
        ml_memory_limit=12288,
        ensemble_size=ensemble_size,
        ensemble_nbest=ensem_nbest,
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
    str_stats = automl.sprint_statistics()
    test_results = automl.cv_results_['mean_test_score']
    time_records = automl.cv_results_['mean_fit_time']
    validation_accuracy = np.max(test_results)
    predictions = automl.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(str_stats)
    print(model_desc)
    print('Validation Accuracy', validation_accuracy)
    print("Test Accuracy", test_accuracy)

    save_path = save_dir + '%s-%d.pkl' % (task_id, rep_id)
    with open(save_path, 'wb') as f:
        stats = [model_desc, str_stats, test_results, time_records, time_limit]
        pickle.dump([validation_accuracy, test_accuracy, stats], f)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    start_id = args.start_id
    rep = args.rep_num
    default_time_cost = args.time_cost

    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    algorithms = ['k_nearest_neighbors', 'libsvm_svc', 'random_forest', 'adaboost']
    if algo_num == 8:
        algorithms = ['lda', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                      'adaboost', 'random_forest', 'extra_trees', 'decision_tree']
    # algorithms.remove('lda')
    dataset_list = dataset_str.split(',')

    for dataset in dataset_list:
        if default_time_cost == -1:
            time_costs = load_hmab_time_costs(start_id, rep, dataset, len(algorithms), trial_num)
            print(time_costs)
            median = time_costs[np.argsort(time_costs)[rep//2]]
            time_costs = [median] * rep
            print(median, time_costs)
        else:
            time_costs = [default_time_cost] * rep

        for run_id in range(start_id, start_id+rep):
            seed = int(seeds[run_id])
            time_cost = time_costs[run_id-start_id]
            evaluate_autosklearn(algorithms, run_id, trial_num, dataset, time_cost, seed=seed)
