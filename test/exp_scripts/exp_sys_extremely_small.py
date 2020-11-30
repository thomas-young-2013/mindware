"""
    This script is used to compare the strategies/algorithms in the FE-HPO selection
    problem and Bayesian optimization based solution (Auto-scikitlearn)
"""
import os
import sys
import shutil
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
parser.add_argument('--enable_meta', type=str, default='false', choices=['true', 'false'])
parser.add_argument('--time_cost', type=int, default=600)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep_num', type=int, default=5)
# choices=['rb', 'alter_hpo', 'fixed', 'plot', 'all', 'ausk', 'combined']
project_dir = './'
save_folder = project_dir + 'data/exp_sys/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def evaluate_sys(run_id, task_type, mth, dataset, ens_method, enable_meta,
                 eval_type='holdout', time_limit=1200, seed=1):
    _task_type = MULTICLASS_CLS if task_type == 'cls' else REGRESSION
    train_data, test_data = load_train_test_data(dataset, task_type=_task_type)
    _enable_meta = True if enable_meta == 'true' else False
    if task_type == 'cls':
        from solnml.estimators import Classifier
        estimator = Classifier(time_limit=time_limit,
                               per_run_time_limit=300,
                               output_dir=save_folder,
                               ensemble_method=ens_method,
                               enable_meta_algorithm_selection=_enable_meta,
                               evaluation=eval_type,
                               metric='bal_acc',
                               include_algorithms=['random_forest'],
                               include_preprocessors=['extra_trees_based_selector',
                                                      'generic_univariate_selector',
                                                      'liblinear_based_selector',
                                                      'percentile_selector'],
                               n_jobs=1)
    else:
        from solnml.estimators import Regressor
        estimator = Regressor(time_limit=time_limit,
                              per_run_time_limit=300,
                              output_dir=save_folder,
                              ensemble_method=ens_method,
                              enable_meta_algorithm_selection=_enable_meta,
                              evaluation=eval_type,
                              metric='mse',
                              include_algorithms=['random_forest'],
                              include_preprocessors=['extra_trees_based_selector_regression',
                                                     'generic_univariate_selector',
                                                     'liblinear_based_selector',
                                                     'percentile_selector_regression'],
                              n_jobs=1)

    start_time = time.time()
    estimator.fit(train_data, opt_strategy=mth, dataset_id=dataset)
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

    save_path = save_folder + 'extremely_small_%s_%s_%s_%s_%d_%d_%d.pkl' % (
        task_type, mth, dataset, enable_meta, time_limit, (ens_method is None), run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, validation_score, test_score, start_time, eval_dict], f)

    # Delete output dir
    shutil.rmtree(os.path.join(estimator.get_output_dir()))


def evaluate_ausk(run_id, task_type, mth, dataset, ens_method, enable_meta,
                  eval_type='holdout', time_limit=1200, seed=1):
    tmp_dir = 'data/exp_sys/ausk_tmp_%s_%s_%s_%d_%d' % (task_type, mth, dataset, time_limit, run_id)
    output_dir = 'data/exp_sys/ausk_output_%s_%s_%s_%d_%d' % (task_type, mth, dataset, time_limit, run_id)
    initial_configs = 25 if enable_meta == 'true' else 0
    if os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
            shutil.rmtree(output_dir)
        except:
            pass

    if task_type == 'cls':
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=int(time_limit),
            per_run_time_limit=300,
            n_jobs=1,
            include_estimators=['random_forest'],
            include_preprocessors=['extra_trees_preproc_for_classification',
                                   'liblinear_svc_preprocessor',
                                   'select_percentile_classification',
                                   'select_rates'],
            ensemble_memory_limit=16384,
            ml_memory_limit=16384,
            ensemble_size=1 if ens_method is None else 50,
            initial_configurations_via_metalearning=initial_configs,
            tmp_folder=tmp_dir,
            output_folder=output_dir,
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
            include_estimators=['random_forest'],
            include_preprocessors=['extra_trees_preproc_for_regression',
                                   'select_percentile_regression',
                                   'select_rates'],
            ensemble_memory_limit=16384,
            ml_memory_limit=16384,
            ensemble_size=1 if ens_method is None else 50,
            initial_configurations_via_metalearning=initial_configs,
            tmp_folder=tmp_dir,
            output_folder=output_dir,
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
    from autosklearn.metrics import make_scorer
    if task_type == 'cls':
        scorer = make_scorer(name='balanced_accuracy', score_func=balanced_accuracy_score)
        score_func = balanced_accuracy_score
    else:
        scorer = make_scorer(name='mean_squared_error', score_func=mean_squared_error, greater_is_better=False)
        score_func = mean_squared_error
    start_time = time.time()
    automl.fit(X.copy(), y.copy(), feat_type=feat_type,
               metric=scorer)
    valid_results = automl.cv_results_['mean_test_score']
    if task_type == 'cls':
        validation_score = np.max(valid_results)
    else:
        valid_results = [ele - valid_results[-1] for ele in valid_results[:-1]]
        validation_score = np.min(valid_results)
    # automl.refit(X.copy(), y.copy())
    predictions = automl.predict(X_test)
    test_score = score_func(y_test, predictions)
    model_desc = automl.show_models()
    str_stats = automl.sprint_statistics()
    result_score = automl.cv_results_['mean_test_score']
    result_time = automl.cv_results_['mean_fit_time']

    print('=' * 10)
    # print(model_desc)
    print(str_stats)
    print('=' * 10)

    print('Validation score', validation_score)
    print('Test score', test_score)
    # print(automl.show_models())
    save_path = save_folder + 'extremely_small_%s_%s_%s_%s_%d_%d_%d.pkl' % (
        task_type, mth, dataset, enable_meta, time_limit, (ens_method is None), run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, validation_score, test_score, start_time, result_score, result_time], f)

    shutil.rmtree(output_dir)
    shutil.rmtree(os.path.join(tmp_dir, '.auto-sklearn'))


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
    enable_meta = args.enable_meta
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
                        evaluate_sys(_id, task_type, method, dataset, ens_method, enable_meta,
                                     eval_type=cv, time_limit=time_cost, seed=seed)
                    elif method in ['ausk']:
                        evaluate_ausk(_id, task_type, method, dataset, ens_method, enable_meta,
                                      eval_type=cv, time_limit=time_cost, seed=seed)
                    else:
                        raise ValueError('Invalid mode: %s!' % method)

    else:
        headers = ['dataset']
        # method_ids = ['fixed', 'alter_hpo', 'rb', 'ausk']
        method_ids = mode.split(',')[1:]
        if len(method_ids) == 0:
            method_ids = ['alter_hpo', 'combined', 'ausk', 'tpot']
        for mth in method_ids:
            headers.extend(['val-%s' % mth, 'test-%s' % mth])
        tbl_data = list()
        for dataset in dataset_list:
            row_data = [dataset]
            for mth in method_ids:
                results = list()
                for run_id in range(rep):
                    if mth == 'tpot':
                        _ens_method = None
                    else:
                        _ens_method = ens_method
                    file_path = save_folder + 'small_%s_%s_%s_%s_%d_%d_%d.pkl' % (
                        task_type, mth, dataset, enable_meta, time_cost, (_ens_method is None), run_id)
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    if mth == 'ausk' and task_type == 'rgs':
                        test_acc = data[2]
                        val_acc = min([ele - 2 for ele in data[4] if ele != 2])
                    elif task_type == 'rgs':
                        val_acc, test_acc = -data[1], data[2]
                        if isinstance(test_acc, list):
                            test_acc = test_acc[-1]
                    else:
                        val_acc, test_acc = data[1], data[2]
                        if isinstance(test_acc, list):
                            test_acc = test_acc[-1]
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
