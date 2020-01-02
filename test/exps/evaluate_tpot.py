import os
import sys
import pickle
import argparse
import numpy as np
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())
from automlToolkit.datasets.utils import load_data, load_train_test_data

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default='diabetes')
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--time_limit', type=int, default=10)
parser.add_argument('--n_job', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)

max_eval_time = 2.5  # That's, 150 seconds.
save_dir = './data/sys_baselines/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def evaluate_tpot(dataset, run_id, time_limit, seed=1, use_fe=True):
    n_job = args.n_job
    # Construct the ML model.
    config = None
    if not use_fe:
        from automlToolkit.utils import classifier_config_dict
        config = classifier_config_dict

    # TODO: tpot's split method needs to be modified. holdout instead of cv.
    automl = TPOTClassifier(config_dict=config, generations=10000, population_size=20,
                            verbosity=2, n_jobs=n_job,
                            max_eval_time_mins=2.5,
                            max_time_mins=time_limit,
                            random_state=seed)

    raw_data, test_raw_data = load_train_test_data(dataset)
    X_train, y_train = raw_data.data
    X_test, y_test = test_raw_data.data
    X_train, y_train = X_train.astype('float64'), y_train.astype('int')
    X_test, y_test = X_test.astype('float64'), y_test.astype('int')
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_hat)
    print("%d-th Evaluation: accuracy score => %.4f" % (run_id, test_accuracy))

    save_path = save_dir + 'tpot-%s-%d-%d.pkl' % (dataset, time_limit, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([test_accuracy], f)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    time_limit = args.time_limit
    start_id = args.start_id
    rep = args.rep_num
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set
    else:
        dataset_list = dataset_str.split(',')

    for dataset in dataset_list:
        for run_id in range(start_id, start_id+rep):
            seed = int(seeds[run_id])
            evaluate_tpot(dataset, run_id, time_limit, seed)
