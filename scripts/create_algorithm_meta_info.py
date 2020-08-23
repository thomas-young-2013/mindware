import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.append(os.getcwd())
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.pipeline.components.classification import add_classifier

from solnml.components.utils.constants import MULTICLASS_CLS, BINARY_CLS, REGRESSION, CLS_TASKS, CATEGORICAL
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from scripts.ausk_udf_models.lightgbm import LightGBM
from scripts.ausk_udf_models.logistic_regression import Logistic_Regression

parser = argparse.ArgumentParser()
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep', type=int, default=3)
parser.add_argument('--datasets', type=str, default='diabetes')
parser.add_argument('--metrics', type=str, default='all')
parser.add_argument('--task', type=str, choices=['reg', 'cls'], default='cls')
parser.add_argument('--algo', type=str, default='all')
parser.add_argument('--time_limit', type=int, default=1200)
args = parser.parse_args()

datasets = args.datasets.split(',')
start_id, rep = args.start_id, args.rep
time_limit = args.time_limit
save_dir = './data/meta_res/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
cls_metrics = ['acc', 'f1', 'auc']
reg_metrics = ['mse', 'r2', 'mae']


def evaluate_ml_algorithm(dataset, algo, run_id, obj_metric, time_limit=600, seed=1, task_type=None):
    if algo == 'lightgbm':
        _algo = ['LightGBM']
        add_classifier(LightGBM)
    elif algo == 'logistic_regression':
        _algo = ['Logistic_Regression']
        add_classifier(Logistic_Regression)
    else:
        _algo = [algo]
    print('EVALUATE-%s-%s-%s: run_id=%d' % (dataset, algo, obj_metric, run_id))
    train_data, test_data = load_train_test_data(dataset, task_type=task_type)
    if task_type in CLS_TASKS:
        task_type = BINARY_CLS if len(set(train_data.data[1])) == 2 else MULTICLASS_CLS
    print(set(train_data.data[1]))

    raw_data, test_raw_data = load_train_test_data(dataset, task_type=MULTICLASS_CLS)
    X, y = raw_data.data
    X_test, y_test = test_raw_data.data
    feat_type = ['Categorical' if _type == CATEGORICAL else 'Numerical'
                 for _type in raw_data.feature_types]
    from autosklearn.metrics import balanced_accuracy as balanced_acc
    automl = AutoSklearnClassifier(
        time_left_for_this_task=int(time_limit),
        per_run_time_limit=180,
        n_jobs=1,
        include_estimators=_algo,
        initial_configurations_via_metalearning=0,
        ensemble_memory_limit=16384,
        ml_memory_limit=16384,
        # tmp_folder='/var/folders/0t/mjph32q55hd10x3qr_kdd2vw0000gn/T/autosklearn_tmp',
        ensemble_size=1,
        seed=int(seed),
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67}
    )
    automl.fit(X.copy(), y.copy(), feat_type=feat_type, metric=balanced_acc)
    model_desc = automl.show_models()
    str_stats = automl.sprint_statistics()
    valid_results = automl.cv_results_['mean_test_score']
    print('Eval num: %d' % (len(valid_results)))

    validation_score = np.max(valid_results)

    # Test performance.
    automl.refit(X.copy(), y.copy())
    predictions = automl.predict(X_test)
    test_score = balanced_accuracy_score(y_test, predictions)

    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(str_stats)
    print(model_desc)
    print('Validation Accuracy:', validation_score)
    print("Test Accuracy      :", test_score)

    save_path = save_dir + '%s-%s-%s-%d-%d.pkl' % (dataset, algo, obj_metric, run_id, time_limit)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, algo, validation_score, test_score, task_type], f)


def check_datasets(datasets, task_type=None):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1, task_type=task_type)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == "__main__":
    algorithms = ['random_forest',
                  'lda',
                  'liblinear_svc',
                  'libsvm_svc'
                  'k_nearest_neighbors',
                  'adaboost',
                  'lightgbm',
                  'gradient_boosting',
                  'qda',
                  'extra_trees']
    task_type = MULTICLASS_CLS
    # if args.task == 'reg':
    #     task_type = REGRESSION
    #     algorithms = ['lightgbm', 'random_forest',
    #                   'libsvm_svr', 'extra_trees',
    #                   'liblinear_svr', 'k_nearest_neighbors',
    #                   'lasso_regression',
    #                   'gradient_boosting', 'adaboost']

    if args.algo != 'all':
        algorithms = args.algo.split(',')

    metrics = cls_metrics if args.task == 'cls' else reg_metrics
    if args.metrics != 'all':
        metrics = args.metrics.split(',')

    check_datasets(datasets, task_type=task_type)
    running_info = list()
    log_filename = 'running-%d.txt' % os.getpid()

    for dataset in datasets:
        for obj_metric in metrics:
            np.random.seed(1)
            seeds = np.random.randint(low=1, high=10000, size=start_id + rep)
            for algo in algorithms:
                for run_id in range(start_id, start_id + rep):
                    seed = seeds[run_id]
                    try:
                        task_id = '%s-%s-%s-%d: %s' % (dataset, algo, obj_metric, run_id, 'success')
                        evaluate_ml_algorithm(dataset, algo, run_id, obj_metric, time_limit=time_limit,
                                              seed=seed, task_type=task_type)
                    except Exception as e:
                        task_id = '%s-%s-%s-%d: %s' % (dataset, algo, obj_metric, run_id, str(e))

                    print(task_id)
                    running_info.append(task_id)
                    with open(save_dir + log_filename, 'a') as f:
                        f.write('\n' + task_id)

    # Write down the error info.
    with open(save_dir + 'failed-%s' % log_filename, 'w') as f:
        f.write('\n'.join(running_info))
