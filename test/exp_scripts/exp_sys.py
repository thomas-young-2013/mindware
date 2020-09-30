"""
    This script is used to compare the strategies/algorithms in the FE-HPO selection
    problem and Bayesian optimization based solution (Auto-scikitlearn)
"""
import os
import sys
import time
import pickle
import argparse
import tabulate
import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

sys.path.append(os.getcwd())
import autosklearn.classification
import autosklearn.regression

from solnml.datasets.utils import load_train_test_data
from solnml.components.utils.constants import CATEGORICAL, MULTICLASS_CLS, REGRESSION

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,vehicle,yeast,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--task_type', type=str, default='cls', choices=['cls', 'rgs'])
parser.add_argument('--mode', type=str, default='alter_hpo')
parser.add_argument('--cv', type=str, choices=['cv', 'holdout', 'partial'], default='holdout')
parser.add_argument('--ens', type=str, default='None')
parser.add_argument('--time_cost', type=int, default=600)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep_num', type=int, default=5)
# choices=['rb', 'alter_hpo', 'fixed', 'plot', 'all', 'ausk', 'combined']
project_dir = './'
save_folder = project_dir + 'data/exp_sys/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def evaluate_sys(run_id, task_type, mth, dataset, ens_method,
                 eval_type='holdout', time_limit=1200, seed=1):
    _task_type = MULTICLASS_CLS if task_type == 'cls' else REGRESSION
    train_data, test_data = load_train_test_data(dataset, task_type=_task_type)

    if task_type == 'cls':
        from solnml.estimators import Classifier
        estimator = Classifier(time_limit=time_limit,
                               per_run_time_limit=300,
                               output_dir=save_folder,
                               ensemble_method=ens_method,
                               evaluation=eval_type,
                               enable_meta_algorithm_selection=False,
                               metric='bal_acc',
                               # include_algorithms=['extra_trees'],
                               n_jobs=1)
    else:
        from solnml.estimators import Regressor
        estimator = Regressor(time_limit=time_limit,
                              per_run_time_limit=300,
                              output_dir=save_folder,
                              ensemble_method=ens_method,
                              evaluation=eval_type,
                              enable_meta_algorithm_selection=False,
                              metric='mse',
                              # include_algorithms=['random_forest'],
                              n_jobs=1)

    start_time = time.time()
    estimator.fit(train_data, opt_strategy=mth)
    pred = estimator.predict(test_data)
    if task_type == 'cls':
        test_score = balanced_accuracy_score(test_data.data[1], pred)
    else:
        test_score = mean_squared_error(test_data.data[1], pred)
    validation_score = estimator._ml_engine.solver.incumbent_perf
    eval_dict = estimator._ml_engine.solver.get_eval_dict()
    print('Run ID         : %d' % run_id)
    print('Dataset        : %s' % dataset)
    print('Val/Test score : %f - %f' % (validation_score, test_score))

    save_path = save_folder + '%s_%s_%s_%d_%d_%d.pkl' % (
        task_type, mth, dataset, time_limit, (ens_method is None), run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, validation_score, test_score, start_time, eval_dict], f)


def evaluate_ausk(run_id, task_type, mth, dataset, ens_method,
                  eval_type='holdout', time_limit=1200, seed=1):
    if task_type == 'cls':
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=int(time_limit),
            per_run_time_limit=300,
            n_jobs=1,
            # include_estimators=[algo],
            ensemble_memory_limit=16384,
            ml_memory_limit=16384,
            ensemble_size=1 if ens_method is None else 50,
            ensemble_nbest=1,
            initial_configurations_via_metalearning=0,
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
            seed=int(seed),
            resampling_strategy='holdout',
            resampling_strategy_arguments={'train_size': 0.67}
        )
    else:
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=int(time_limit),
            per_run_time_limit=300,
            n_jobs=1,
            # include_estimators=[algo],
            ensemble_memory_limit=16384,
            ml_memory_limit=16384,
            ensemble_size=1 if ens_method is None else 50,
            ensemble_nbest=1,
            initial_configurations_via_metalearning=0,
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
            seed=int(seed),
            resampling_strategy='holdout',
            resampling_strategy_arguments={'train_size': 0.67}
        )

    print(automl)
    _task_type = MULTICLASS_CLS if task_type == 'cls' else REGRESSION
    train_data, test_data = load_train_test_data(dataset, task_type=_task_type)
    X, y = train_data.data
    X_test, y_test = test_data.data
    feat_type = ['Categorical' if _type == CATEGORICAL else 'Numerical'
                 for _type in train_data.feature_types]
    from sklearn.metrics import balanced_accuracy_score
    from autosklearn.metrics import make_scorer
    start_time = time.time()
    automl.fit(X.copy(), y.copy(), feat_type=feat_type,
               metric=make_scorer(name='bal_acc', score_func=balanced_accuracy_score))
    valid_results = automl.cv_results_['mean_test_score']
    validation_score = np.max(valid_results)
    # automl.refit(X.copy(), y.copy())
    predictions = automl.predict(X_test)
    test_score = balanced_accuracy_score(y_test, predictions)
    model_desc = automl.show_models()
    str_stats = automl.sprint_statistics()
    result_score = automl.cv_results_['mean_test_score']
    result_time = automl.cv_results_['mean_fit_time']

    print('=' * 10)
    print(model_desc)
    print(str_stats)
    print('=' * 10)

    print('Validation score', validation_score)
    print('Test score', test_score)
    print(automl.show_models())
    save_path = save_folder + '%s_%s_%s_%d_%d_%d.pkl' % (
        task_type, mth, dataset, time_limit, (ens_method is None), run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, validation_score, test_score, start_time, result_score, result_time], f)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    time_cost = args.time_cost
    mode = args.mode
    task_type = args.task_type
    ens_method = args.ens
    if ens_method == 'None':
        ens_method = None
    cv = args.cv
    np.random.seed(1)
    rep = args.rep_num
    start_id = args.start_id
    seeds = np.random.randint(low=1, high=10000, size=start_id + rep)
    dataset_list = dataset_str.split(',')

    if not mode.startswith('plot'):
        if mode == 'all':
            methods = ['rb', 'fixed', 'alter_hpo']
        else:
            methods = [mode]

        for dataset in dataset_list:

            for method in methods:
                for _id in range(start_id, start_id + rep):
                    seed = seeds[_id]
                    print('Running %s with %d-th seed' % (dataset, _id + 1))
                    if method in ['rb', 'fixed', 'alter_hpo', 'combined', 'rb_hpo']:
                        evaluate_sys(_id, task_type, method, dataset, ens_method,
                                     eval_type=cv, time_limit=time_cost, seed=seed)
                    elif method in ['ausk']:
                        evaluate_ausk(_id, task_type, method, dataset, ens_method,
                                      eval_type=cv, time_limit=time_cost, seed=seed)
                    else:
                        raise ValueError('Invalid mode: %s!' % method)

    else:
        headers = ['dataset']
        # method_ids = ['fixed', 'alter_hpo', 'rb', 'ausk']
        method_ids = mode.split(',')[1:]
        if len(method_ids) == 0:
            method_ids = ['combined', 'ausk']
        for mth in method_ids:
            headers.extend(['val-%s' % mth, 'test-%s' % mth])
        tbl_data = list()
        for dataset in dataset_list:
            row_data = [dataset]
            for mth in method_ids:
                results = list()
                for run_id in range(rep):
                    file_path = save_folder + '%s_%s_%d_%d_%s.pkl' % (mth, dataset, time_cost, run_id, algo)
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    val_acc, test_acc = data[1], data[2]
                    results.append([val_acc, test_acc])
                print(mth, results)
                if len(results) == rep:
                    results = np.array(results)
                    # print(mth, results)
                    stats_ = zip(np.mean(results, axis=0), np.std(results, axis=0))
                    string = ''
                    for mean_t, std_t in stats_:
                        string += u'%.3f\u00B1%.3f |' % (mean_t, std_t)
                    print(dataset, mth, '=' * 30)
                    print('%s-%s: mean\u00B1std' % (dataset, mth), string)
                    print('%s-%s: median' % (dataset, mth), np.median(results, axis=0))

                    for idx in range(results.shape[1]):
                        vals = results[:, idx]
                        median = np.median(vals)
                        if median == 0.:
                            row_data.append('-')
                        else:
                            row_data.append(u'%.4f' % median)
                else:
                    row_data.extend(['-'] * 2)

            tbl_data.append(row_data)
        print(tabulate.tabulate(tbl_data, headers, tablefmt='github'))
