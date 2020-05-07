import os
import pickle
import numpy as np
from automlToolkit.datasets.utils import calculate_metafeatures


def get_feature_vector(dataset, data_dir='./', task_type=None):
    feature_dict = calculate_metafeatures(dataset, data_dir, task_type=task_type)
    sorted_keys = sorted(feature_dict.keys())
    return [feature_dict[key] for key in sorted_keys]


def fetch_algorithm_runs(meta_dir, dataset, metric, total_resource, rep,
                         buildin_algorithms):
    median_score = list()
    for algo in buildin_algorithms:
        scores = list()
        for run_id in range(rep):
            save_path = meta_dir + '%s-%s-%s-%d-%d.pkl' % (dataset, algo, metric, run_id, total_resource)
            if not os.path.exists(save_path):
                continue

            with open(save_path, 'rb') as f:
                res = pickle.load(f)
                scores.append(res[2])

        if len(scores) == 3:
            median_score.append(np.median(scores))
        else:
            median_score.append(-1)
    return median_score


def prepare_meta_dataset(meta_dir, metric, total_resource, rep,
                         buildin_datasets, buildin_algorithms, task_type=None):
    X, Y = list(), list()
    sorted_keys = None
    include_datasets = list()
    for _dataset in buildin_datasets:
        print('Processing', _dataset)
        # Calculate metafeature for datasets.
        try:
            feature_dict = calculate_metafeatures(_dataset, task_type=task_type)
        except Exception as e:
            print(e)
            continue
        if sorted_keys is None:
            sorted_keys = sorted(feature_dict.keys())
        meta_instance = [feature_dict[key] for key in sorted_keys]
        X.append(meta_instance)

        # Load partial relationship between algorithms.
        scores = fetch_algorithm_runs(meta_dir, _dataset, metric, total_resource, rep, buildin_algorithms)
        Y.append(scores)
        include_datasets.append(_dataset)

    return X, Y, include_datasets
