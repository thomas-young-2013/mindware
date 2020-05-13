import os
import sys
import time
import pickle
import argparse
import numpy as np
from tabulate import tabulate
from sklearn.metrics import make_scorer
import autosklearn.classification

sys.path.append(os.getcwd())
from solnml.components.utils.constants import CATEGORICAL
from solnml.datasets.utils import load_train_test_data
from solnml.bandits.first_layer_bandit import FirstLayerBandit
from solnml.components.metrics.cls_metrics import balanced_accuracy
from solnml.components.utils.constants import MULTICLASS_CLS, BINARY_CLS
from solnml.components.meta_learning.algorithm_recomendation.algorithm_advisor import AlgorithmAdvisor
from solnml.utils.functions import is_unbalanced_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='pc4')
parser.add_argument('--mode', type=str, choices=['hmab', 'plot'], default='plot')
parser.add_argument('--algo_num', type=int, default=15)
parser.add_argument('--time_cost', type=int, default=600)
parser.add_argument('--trial_num', type=int, default=150)
parser.add_argument('--rep_num', type=int, default=10)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)


project_dir = './data/meta_exp/'
per_run_time_limit = 120
opt_algo = 'fixed'
hmab_flag = 'hmab_pipeline_meta'
if not os.path.exists(project_dir):
    os.makedirs(project_dir)


def evaluate_hmab(algorithms, dataset, run_id, trial_num, seed, time_limit=1200):
    print('%s-%s-%d: %d' % (hmab_flag, dataset, run_id, time_limit))
    exclude_datasets = ['gina_prior2', 'pc2', 'abalone', 'wind', 'waveform-5000(2)',
                        'page-blocks(1)', 'winequality_white', 'pollen']
    alad = AlgorithmAdvisor(task_type=MULTICLASS_CLS, n_algorithm=9,
                            metric='bal_acc', exclude_datasets=exclude_datasets)
    n_algo = 5
    assert dataset in exclude_datasets
    meta_infos = alad.fit_meta_learner()
    assert dataset not in meta_infos
    model_candidates = alad.fetch_algorithm_set(dataset)
    include_models = list()
    print(model_candidates)
    for algo in model_candidates:
        if algo in algorithms and len(include_models) < n_algo:
            include_models.append(algo)
    print('After algorithm recommendation', include_models)
    # if dataset in ['page-blocks(1)', 'pc2']:
    #     include_models = ['libsvm_svc']
    # elif dataset == 'winequality_white':
    #     include_models = ['liblinear_svc']
    # else:
    #     pass
    _start_time = time.time()
    train_data, test_data = load_train_test_data(dataset, task_type=MULTICLASS_CLS)
    cls_task_type = BINARY_CLS if len(set(train_data.data[1])) == 2 else MULTICLASS_CLS
    balanced_acc_metric = make_scorer(balanced_accuracy)

    if is_unbalanced_dataset(train_data):
        from solnml.components.feature_engineering.transformations.preprocessor.to_balanced import DataBalancer
        train_data = DataBalancer().operate(train_data)
    bandit = FirstLayerBandit(cls_task_type, trial_num, include_models, train_data,
                              output_dir='logs',
                              per_run_time_limit=per_run_time_limit,
                              dataset_name=dataset,
                              ensemble_size=50,
                              inner_opt_algorithm=opt_algo,
                              metric=balanced_acc_metric,
                              fe_algo='bo',
                              seed=seed,
                              time_limit=time_limit,
                              eval_type='holdout')
    bandit.optimize()
    time_taken = time.time() - _start_time
    model_desc = [bandit.nbest_algo_ids, bandit.optimal_algo_id, bandit.final_rewards, bandit.action_sequence]

    validation_accuracy = np.max(bandit.final_rewards)
    best_pred = bandit._best_predict(test_data)
    test_accuracy = balanced_accuracy(test_data.data[1], best_pred)

    bandit.refit()
    es_pred = bandit._es_predict(test_data)
    test_accuracy_with_ens = balanced_accuracy(test_data.data[1], es_pred)

    data = [dataset, validation_accuracy, test_accuracy, test_accuracy_with_ens, time_taken, model_desc]
    print(model_desc)
    print(data)

    save_path = project_dir + '%s_%s_%s_%d_%d_%d_%d_%d.pkl' % (
        hmab_flag, opt_algo, dataset, trial_num, len(algorithms), seed, run_id, time_limit)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def evaluate_autosklearn(algorithms, dataset, run_id, trial_num, seed, time_limit=1200):
    print('AUSK-%s-%d: %d' % (dataset, run_id, time_limit))
    ausk_flag = 'ausk_full'
    if ausk_flag == 'ausk_alad':
        alad = AlgorithmAdvisor(task_type=MULTICLASS_CLS, n_algorithm=9, metric='acc')
        meta_infos = alad.fit_meta_learner()
        assert dataset not in meta_infos
        model_candidates = alad.fetch_algorithm_set(dataset)
        include_models = list()
        print(model_candidates)
        for algo in model_candidates:
            if algo in algorithms and len(include_models) < 3:
                include_models.append(algo)
        print('After algorithm recommendation', include_models)
        n_config_meta_learning = 0
        ensemble_size = 1
    elif ausk_flag == 'ausk_no_meta':
        include_models = algorithms
        n_config_meta_learning = 25
        ensemble_size = 1
    elif ausk_flag == 'ausk_full':
        include_models = algorithms
        n_config_meta_learning = 25
        ensemble_size = 50
    else:
        include_models = algorithms
        n_config_meta_learning = 0
        ensemble_size = 1

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_limit,
        per_run_time_limit=per_run_time_limit,
        include_preprocessors=None,
        exclude_preprocessors=None,
        n_jobs=1,
        include_estimators=include_models,
        ensemble_memory_limit=8192,
        ml_memory_limit=8192,
        ensemble_size=ensemble_size,
        ensemble_nbest=ensemble_size,
        initial_configurations_via_metalearning=n_config_meta_learning,
        seed=int(seed),
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67}
    )
    print(automl)

    train_data, test_data = load_train_test_data(dataset, task_type=MULTICLASS_CLS)
    X, y = train_data.data
    feat_type = ['Categorical' if _type == CATEGORICAL else 'Numerical'
                 for _type in train_data.feature_types]
    from autosklearn.metrics import balanced_accuracy
    automl.fit(X.copy(), y.copy(), metric=balanced_accuracy, feat_type=feat_type)
    model_desc = automl.show_models()
    print(model_desc)
    val_result = np.max(automl.cv_results_['mean_test_score'])
    print('Trial number', len(automl.cv_results_['mean_test_score']))
    print('Best validation accuracy', val_result)

    X_test, y_test = test_data.data
    automl.refit(X.copy(), y.copy())
    y_pred = automl.predict(X_test)
    metric = balanced_accuracy
    test_result = metric(y_test, y_pred)
    print('Test accuracy', test_result)
    save_path = project_dir + '%s_%s_%d_%d_%d_%d_%d.pkl' % (
        ausk_flag, dataset, trial_num, len(algorithms), seed, run_id, time_limit)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, val_result, test_result, model_desc], f)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo_num = args.algo_num
    trial_num = args.trial_num
    modes = args.mode.split(',')
    rep = args.rep_num
    start_id = args.start_id
    time_limit = args.time_cost

    # Prepare random seeds.
    np.random.seed(args.seed)
    seeds = np.random.randint(low=1, high=10000, size=start_id + args.rep_num)

    algorithms = ['adaboost', 'random_forest',
                  'libsvm_svc', 'sgd',
                  'extra_trees', 'decision_tree',
                  'liblinear_svc', 'k_nearest_neighbors',
                  'passive_aggressive', 'xgradient_boosting',
                  'lda', 'qda',
                  'multinomial_nb', 'gaussian_nb', 'bernoulli_nb'
                  ]

    dataset_list = dataset_str.split(',')

    for mode in modes:
        if mode != 'plot':
            for dataset in dataset_list:
                for run_id in range(start_id, start_id + rep):
                    seed = int(seeds[run_id])
                    if mode == 'hmab':
                        evaluate_hmab(algorithms, dataset, run_id, trial_num, seed, time_limit=time_limit)
                    else:
                        evaluate_autosklearn(algorithms, dataset, run_id, trial_num, seed, time_limit=time_limit)
        else:
            headers = ['dataset']
            method_ids = ['hmab_pipeline_meta_fixed', 'ausk_full']
            for mth in method_ids:
                headers.extend(['val-%s' % mth, 'test-%s' % mth])

            tbl_data = list()
            for dataset in dataset_list:
                row_data = [dataset]
                for mth in method_ids:
                    results = list()
                    for run_id in range(rep):
                        seed = seeds[run_id]
                        time_t = time_limit
                        file_path = project_dir + '%s_%s_%d_%d_%d_%d_%d.pkl' % (
                            mth, dataset, trial_num, len(algorithms), seed, run_id, time_t)
                        if not os.path.exists(file_path):
                            continue
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        if mth.startswith('hmab'):
                            val_acc, test_acc = data[1], data[3]
                        else:
                            val_acc, test_acc = data[1], data[2]
                        results.append([val_acc, test_acc])
                        # if mth.startswith('ausk'):
                        #     print('='*10)
                        #     print(val_acc, test_acc)
                        #     print(data[3])
                        #     print('='*10)

                    if len(results) == rep:
                        results = np.array(results)
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
            print(tabulate(tbl_data, headers, tablefmt='github'))
