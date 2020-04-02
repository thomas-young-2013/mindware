import os
import sys
import time
import pickle
import argparse
import numpy as np
import autosklearn.classification
from tabulate import tabulate
sys.path.append(os.getcwd())
from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.evaluators.cls_evaluator import ClassificationEvaluator, fetch_predict_estimator

parser = argparse.ArgumentParser()
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--time_limit', type=int, default=1200)
parser.add_argument('--datasets', type=str, default='dataset_small')
parser.add_argument('--mths', type=str, default='ausk,hmab')
args = parser.parse_args()

# dataset_large = 'credit,diabetes,pc4,sick,spectf,splice,waveform,' \
#                 'winequality_red,winequality_white,ionosphere,amazon_employee,' \
#                 'lymphography,messidor_features,spambase,ap_omentum_ovary,a9a'
dataset_small = 'messidor_features,lymphography,winequality_red,winequality_white,credit,' \
                'ionosphere,splice,diabetes,pc4,spectf,spambase,amazon_employee'

if args.datasets == 'all':
    datasets = dataset_small.split(',')
else:
    datasets = args.datasets.split(',')
time_limit = args.time_limit
mths = args.mths.split(',')
start_id, rep = args.start_id, args.rep
save_dir = './data/eval_exps/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def evaluate_ausk_fe(dataset, time_limit, run_id, seed):
    print('[Run ID: %d] Start to Evaluate' % run_id, dataset, 'Budget', time_limit)
    from automlToolkit.utils.models.default_random_forest import DefaultRandomForest
    # Add random forest classifier (with default hyperparameter) component to auto-sklearn.
    autosklearn.pipeline.components.classification.add_classifier(DefaultRandomForest)
    include_models = ['DefaultRandomForest']

    # Construct the ML model.
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_limit,
        include_preprocessors=None,
        n_jobs=1,
        include_estimators=include_models,
        ensemble_memory_limit=8192,
        ml_memory_limit=8192,
        ensemble_size=1,
        initial_configurations_via_metalearning=0,
        per_run_time_limit=300,
        seed=int(seed),
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67}
        # resampling_strategy='cv',
        # resampling_strategy_arguments={'folds': 5}
    )
    print(automl)

    train_data, test_data = load_train_test_data(dataset)

    X, y = train_data.data
    X_test, y_test = test_data.data

    from autosklearn.metrics import balanced_accuracy
    automl.fit(X.copy(), y.copy(), metric=balanced_accuracy)
    model_desc = automl.show_models()
    print(model_desc)

    # print(automl.cv_results_)
    val_result = np.max(automl.cv_results_['mean_test_score'])
    print('Best validation accuracy', val_result)

    # automl.refit(X.copy(), y.copy())
    test_result = automl.score(X_test, y_test)
    print('Test accuracy', test_result)

    save_path = save_dir + 'ausk_fe_%s_%d_%d.pkl' % (dataset, time_limit, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, val_result, test_result, model_desc], f)


def evaluate_evaluation_based_fe(dataset, time_limit, run_id, seed):
    from automlToolkit.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer

    # Prepare the configuration for random forest.
    from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
    from autosklearn.pipeline.components.classification.random_forest import RandomForest
    cs = RandomForest.get_hyperparameter_search_space()
    clf_hp = UnParametrizedHyperparameter("estimator", 'random_forest')
    cs.add_hyperparameter(clf_hp)
    print(cs.get_default_configuration())
    """
    Configuration:
      bootstrap, Value: 'True'
      criterion, Value: 'gini'
      estimator, Constant: 'random_forest'
      max_depth, Constant: 'None'
      max_features, Value: 0.5
      max_leaf_nodes, Constant: 'None'
      min_impurity_decrease, Constant: 0.0
      min_samples_leaf, Value: 1
      min_samples_split, Value: 2
      min_weight_fraction_leaf, Constant: 0.0
      n_estimators, Constant: 100
    """
    evaluator = ClassificationEvaluator(cs.get_default_configuration(), name='fe', seed=seed,
                                        resampling_strategy='holdout')

    train_data, test_data = load_train_test_data(dataset)
    optimizer = EvaluationBasedOptimizer('classification', train_data, evaluator,
                                         'random_forest', 300, 10000,
                                         seed, trans_set=None)

    _start_time = time.time()
    _iter_id = 0
    while True:
        if time.time() > _start_time + time_limit or optimizer.early_stopped_flag:
            break
        score, iteration_cost, inc = optimizer.iterate()
        print('%d - %.4f' % (_iter_id, score))
        _iter_id += 1

    final_train_data = optimizer.apply(train_data, optimizer.incumbent)
    val_score = evaluator(None, data_node=final_train_data)
    print('==> Best validation score', val_score, score)

    final_test_data = optimizer.apply(test_data, optimizer.incumbent)
    X_train, y_train = final_train_data.data
    clf = fetch_predict_estimator(cs.get_default_configuration(), X_train, y_train)
    X_test, y_test = final_test_data.data
    y_pred = clf.predict(X_test)

    from automlToolkit.components.metrics.cls_metrics import balanced_accuracy
    test_score = balanced_accuracy(y_test, y_pred)
    print('==> Test score', test_score)

    save_path = save_dir + 'hmab_fe_%s_%d_%d.pkl' % (dataset, time_limit, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, val_score, test_score], f)


if __name__ == "__main__":
    if mths[0] != 'plot':
        for dataset in datasets:
            # Prepare random seeds.
            np.random.seed(1)
            seeds = np.random.randint(low=1, high=10000, size=start_id + rep)
            for run_id in range(start_id, start_id+rep):
                seed = seeds[run_id]
                for mth in mths:
                    if mth == 'ausk':
                        evaluate_ausk_fe(dataset, time_limit, run_id, seed)
                    else:
                        evaluate_evaluation_based_fe(dataset, time_limit, run_id, seed)
    else:
        headers = ['dataset']
        method_ids = ['hmab', 'ausk']
        for mth in method_ids:
            headers.extend(['val-%s' % mth, 'test-%s' % mth])

        tbl_data = list()
        for dataset in datasets:
            row_data = [dataset]
            for mth in method_ids:
                results = list()
                for run_id in range(rep):
                    task_id = '%s_fe_%s_%d_%d.pkl' % (mth, dataset, time_limit, run_id)
                    file_path = save_dir + task_id
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    val_acc, test_acc = data[1], data[2]
                    results.append([val_acc, test_acc])
                    print(data)
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


"""
 "[(1.000000, SimpleClassificationPipeline({
 'classifier:DefaultRandomForest:min_samples_leaf': 1, 
 'classifier:DefaultRandomForest:n_estimators': 100, 
 'classifier:DefaultRandomForest:min_weight_fraction_leaf': 0.0},
 'classifier:DefaultRandomForest:min_samples_split': 2, 
 'classifier:DefaultRandomForest:max_leaf_nodes': 'None', 
 'classifier:DefaultRandomForest:max_depth': 'None', 
 'classifier:__choice__': 'DefaultRandomForest', 
 'classifier:DefaultRandomForest:criterion': 'gini', 
 'classifier:DefaultRandomForest:bootstrap': 'True', 
 'classifier:DefaultRandomForest:max_features': 'auto',    
 'classifier:DefaultRandomForest:min_impurity_decrease': 0.0, 
    
 'preprocessor:extra_trees_preproc_for_classification:max_depth': 'None', 
 'preprocessor:extra_trees_preproc_for_classification:max_leaf_nodes': 'None', 
 'preprocessor:extra_trees_preproc_for_classification:criterion': 'entropy', 
 'preprocessor:__choice__': 'extra_trees_preproc_for_classification', 
 'preprocessor:extra_trees_preproc_for_classification:n_estimators': 100, 
 'preprocessor:extra_trees_preproc_for_classification:min_samples_leaf': 4, 
 'preprocessor:extra_trees_preproc_for_classification:bootstrap': 'True', 
 'preprocessor:extra_trees_preproc_for_classification:min_samples_split': 2, 
 'preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf': 0.0, 
 'preprocessor:extra_trees_preproc_for_classification:max_features': 0.854608381185211, 
 'preprocessor:extra_trees_preproc_for_classification:min_impurity_decrease': 0.0, 
  
 'categorical_encoding:__choice__': 'one_hot_encoding', 
 'rescaling:__choice__': 'robust_scaler', 
 'rescaling:robust_scaler:q_min': 0.17423633909007513, 
 'rescaling:robust_scaler:q_max': 0.9277153163496668, 
 'imputation:strategy': 'median', 
 'balancing:strategy': 'weighting',  
 'categorical_encoding:one_hot_encoding:use_minimum_fraction': 'False', 
 
 \ndataset_properties={\n  'multiclass': False,\n  'task': 1,\n  'multilabel': False,\n  'signed': False,\n  'sparse': False,\n  'target_type': 'classification'})),\n]"]

"""
