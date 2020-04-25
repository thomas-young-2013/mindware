import os
import pickle
import numpy as np
from automlToolkit.datasets.utils import calculate_metafeatures

buildin_datasets = ['diabetes', 'fri_c1', 'ionosphere']
buildin_algorithns = ['lightgbm', 'random_forest', 'libsvm_svc', 'extra_trees', 'liblinear_svc',
                      'k_nearest_neighbors', 'logistic_regression', 'gradient_boosting', 'adaboost']
meta_dir = './data/meta_res/'
seeds = [236, 5193, 906]


def get_feature_vector(dataset, data_dir='./'):
    feature_dict = calculate_metafeatures(dataset, data_dir)
    sorted_keys = sorted(feature_dict.keys())
    return [feature_dict[key] for key in sorted_keys]


def fetch_algorithm_runs(dataset, total_resource=20, rep=3):
    median_score = list()
    for algo in buildin_algorithns:
        scores = list()
        for run_id in range(rep):
            save_path = meta_dir + '%s_%s_%d_%d_%d.pkl' % (dataset, algo, run_id, seeds[run_id], total_resource)
            if not os.path.exists(save_path):
                continue

            with open(save_path, 'rb') as f:
                res = pickle.load(f)
                scores.append(res[2])

        if len(scores) == rep:
            median_score.append(np.median(scores))
    return median_score


def prepare_meta_dataset(datasets=None):
    X, Y = list(), list()
    sorted_keys = None
    for _dataset in buildin_datasets:
        # Calculate metafeature for datasets.
        feature_dict = calculate_metafeatures(_dataset)
        if sorted_keys is None:
            sorted_keys = sorted(feature_dict.keys())
        meta_instance = [feature_dict[key] for key in sorted_keys]
        X.append(meta_instance)

        # Load partial relationship between algorithms.
        scores = fetch_algorithm_runs(_dataset)
        Y.append(scores)

    return buildin_algorithns, X, Y
