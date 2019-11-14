import os
import sys
import pickle
import argparse
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
from tpot import TPOTClassifier
from tabulate import tabulate

proj_dir = '/home/thomas/PycharmProjects/Feature-Engineering/'
if not os.path.isdir(proj_dir):
    proj_dir = './'

sys.path.append(proj_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--rep', type=int, default=3)
parser.add_argument('--time_limit', type=int, default=10)
parser.add_argument('--n_job', type=int, default=2)
parser.add_argument('--datasets', type=str, default='diabetes')
args = parser.parse_args()

from data_loader import load_data


def exp_trial(dataset, time_limit, fe=True):
    n_job = args.n_job
    # Construct the ML model.

    if fe:
        config = None
    else:
        from utils.tpot_config import classifier_config_dict
        config = classifier_config_dict

    def get_automl(seed):
        automl = TPOTClassifier(config_dict=config, generations=10000, population_size=20,
                                verbosity=2, n_jobs=n_job,
                                max_eval_time_mins=10,
                                max_time_mins=time_limit,
                                random_state=seed)
        print(automl)
        return automl

    results = list()
    np.random.seed(42)
    for rep_id in range(args.rep):
        X, y, _ = load_data(dataset)
        X = X.astype('float64')
        y = y.astype('int')
        seed = np.random.randint(10000000)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=seed)

        automl = get_automl(seed)
        automl.fit(X_train, y_train)
        y_hat = automl.predict(X_test)

        acc = sklearn.metrics.accuracy_score(y_test, y_hat)
        results.append(acc)
        print("Accuracy score", acc)
    return np.mean(results)


def evalaute_tpot_fe():
    datasets = args.datasets.split(',')
    time_limit = args.time_limit

    headers = ['dataset', 'perf1', 'perf2']
    save_template = proj_dir + 'data/tpot_fe_result_%s_%d.pkl'

    for dataset in datasets:
        data = list()
        save_path = save_template % (dataset, time_limit)
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                headers, data = pickle.load(f)
        else:
            res1 = exp_trial(dataset, time_limit)
            res2 = exp_trial(dataset, time_limit, fe=False)
            data.append([dataset, res1, res2])
            with open(save_path, 'wb') as f:
                pickle.dump([headers, data], f)

        print(tabulate(data, headers, tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    evalaute_tpot_fe()
