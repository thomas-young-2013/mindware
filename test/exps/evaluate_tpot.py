import os
import sys
import pickle
import argparse
import numpy as np
import time
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

sys.path.append(os.getcwd())
from tpot import TPOTClassifier, TPOTRegressor
from solnml.datasets.utils import load_data, load_train_test_data
from solnml.components.utils.constants import MULTICLASS_CLS, REGRESSION

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default='diabetes')
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--time_cost', type=int, default=600)
parser.add_argument('--n_job', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--task_type', type=str, default='cls', choices=['cls', 'rgs'])
parser.add_argument('--space_type', type=str, default='large', choices=['large', 'small', 'extremelly_small'])

max_eval_time = 5
save_dir = './data/exp_sys/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def evaluate_tpot(dataset, task_type, run_id, time_limit, seed=1, use_fe=True):
    n_job = args.n_job
    # Construct the ML model.
    if not use_fe:
        from solnml.utils.tpot_config import classifier_config_dict
        config = classifier_config_dict

    _task_type = MULTICLASS_CLS if task_type == 'cls' else REGRESSION
    if space_type == 'large':
        from tpot.config.classifier import classifier_config_dict
        config_dict = classifier_config_dict
    elif space_type == 'small':
        from tpot.config.classifier_small import classifier_config_dict
        config_dict = classifier_config_dict
    else:
        from tpot.config.classifier_extremelly_small import classifier_config_dict
        config_dict = classifier_config_dict

    if task_type == 'cls':
        automl = TPOTClassifier(config_dict=config_dict, generations=10000, population_size=20,
                                verbosity=2, n_jobs=n_job, cv=0.2,
                                scoring='balanced_accuracy',
                                max_eval_time_mins=max_eval_time,
                                max_time_mins=int(time_limit / 60),
                                random_state=seed)
    else:
        automl = TPOTRegressor(config_dict=None, generations=10000, population_size=20,
                               verbosity=2, n_jobs=n_job, cv=0.2,
                               scoring='neg_mean_squared_error',
                               max_eval_time_mins=max_eval_time,
                               max_time_mins=int(time_limit / 60),
                               random_state=seed)

    raw_data, test_raw_data = load_train_test_data(dataset, task_type=_task_type)
    X_train, y_train = raw_data.data
    X_test, y_test = test_raw_data.data
    X_train, y_train = X_train.astype('float64'), y_train.astype('int')
    X_test, y_test = X_test.astype('float64'), y_test.astype('int')
    start_time = time.time()
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    pareto_front = automl._pareto_front

    if task_type == 'cls':
        score_func = balanced_accuracy_score
    else:
        score_func = mean_squared_error

    valid_score = max([pareto_front.keys[x].wvalues[1] for x in range(len(pareto_front.keys))])
    test_score = score_func(y_test, y_hat)
    print('Run ID         : %d' % run_id)
    print('Dataset        : %s' % dataset)
    print('Val/Test score : %f - %f' % (valid_score, test_score))
    scores = automl.scores
    times = automl.times
    save_path = save_dir + '%s_tpot_%s_false_%d_1_%d.pkl' % (task_type, dataset, time_limit, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, valid_score, test_score, times, scores, start_time], f)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    time_limit = args.time_cost
    start_id = args.start_id
    rep = args.rep_num
    task_type = args.task_type
    space_type = args.space_type
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set
    else:
        dataset_list = dataset_str.split(',')

    for dataset in dataset_list:
        for run_id in range(start_id, start_id + rep):
            seed = int(seeds[run_id])
            evaluate_tpot(dataset, task_type, run_id, time_limit, seed)
