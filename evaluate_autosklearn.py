import os
import sys
import pickle
import argparse
import numpy as np
import sklearn.metrics
import sklearn.datasets
from tabulate import tabulate
import sklearn.model_selection
import autosklearn.classification

proj_dir = '/home/thomas/PycharmProjects/Feature-Engineering/'
if not os.path.exists(proj_dir):
    proj_dir = './'
sys.path.append(proj_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--rep', type=int, default=3)
parser.add_argument('--time_limit', type=int, default=600)
parser.add_argument('--n_job', type=int, default=1)
parser.add_argument('--datasets', type=str, default='diabetes')
args = parser.parse_args()

from data_loader import load_data


def exp_trial(dataset, time_limit, fe=True, ensb_size=1, include_models=None):
    # Disable meta-learning.
    initial_config_num = 0
    # Include ML models.
    # includes = ['libsvm_svc', 'random_forest',
    # 'adaboost', 'gradient_boosting', 'decision_tree', 'k_nearest_neighbors']
    n_job = args.n_job
    includes_fe = ['no_preprocessing'] if fe is False else None

    # Construct the ML model.
    def get_automl(seed):
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time_limit,
            include_preprocessors=includes_fe,
            n_jobs=n_job,
            include_estimators=include_models,
            ensemble_memory_limit=8192,
            ensemble_size=ensb_size,
            initial_configurations_via_metalearning=initial_config_num,
            seed=seed,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5}
        )
        print(automl)
        return automl

    results = list()
    np.random.seed(42)
    for rep_id in range(args.rep):
        X, y, _ = load_data(dataset)
        seed = np.random.randint(10000000)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=seed)

        automl = get_automl(seed)
        automl.fit(X_train.copy(), y_train.copy())
        print(automl.show_models())
        automl.refit(X_train.copy(), y_train.copy())

        y_hat = automl.predict(X_train)
        val_acc = sklearn.metrics.accuracy_score(y_train, y_hat)
        print("Validation Accuracy - %d" % rep_id, val_acc)

        y_hat = automl.predict(X_test)
        test_acc = sklearn.metrics.accuracy_score(y_test, y_hat)
        print("Test Accuracy - %d" % rep_id, test_acc)

        results.append([val_acc, test_acc])

    print('Mean Accuracy', np.mean(results, axis=0))

    return np.mean(results, axis=0)


def evalaute_autosklearn_fe():
    datasets = args.datasets.split(',')
    time_limit = args.time_limit
    headers = ['dataset', 'val', 'test']
    save_template = proj_dir + 'data/autosklearn_fe_result_%s_%d.pkl'

    for dataset in datasets:
        data = list()
        save_path = save_template % (dataset, time_limit)
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                headers, data = pickle.load(f)
        else:
            res1 = exp_trial(dataset, time_limit, fe=True)
            res2 = exp_trial(dataset, time_limit, fe=False)
            # res3 = exp_trial(dataset, time_limit, fe=True)
            # res4 = exp_trial(dataset, time_limit, fe=False)
            # data.append([dataset, res1, res2, res3, res4])
            data.append(['fe1', res1[0], res1[1]])
            data.append(['fe0', res2[0], res2[1]])
            with open(save_path, 'wb') as f:
                pickle.dump([headers, data], f)

        print(tabulate(data, headers, tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    evalaute_autosklearn_fe()
