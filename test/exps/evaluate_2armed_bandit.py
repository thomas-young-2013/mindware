import os
import sys
import time
import pickle
import argparse
import numpy as np
import autosklearn.classification
from tabulate import tabulate
sys.path.append(os.getcwd())
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator, fetch_predict_estimator
from solnml.components.metrics.cls_metrics import balanced_accuracy
from solnml.bandits.second_layer_bandit import SecondLayerBandit
from solnml.components.utils.constants import CATEGORICAL
from solnml.datasets.utils import load_train_test_data

parser = argparse.ArgumentParser()
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--time_limit', type=int, default=1200)
parser.add_argument('--datasets', type=str, default='dataset_small')
parser.add_argument('--mths', type=str, default='ausk,hmab')
parser.add_argument('--algo', type=str, default='random_forest')
args = parser.parse_args()

dataset_small = 'messidor_features,lymphography,winequality_red,winequality_white,credit,' \
                'ionosphere,splice,diabetes,pc4,spectf,spambase,amazon_employee'

if args.datasets == 'all':
    datasets = dataset_small.split(',')
else:
    datasets = args.datasets.split(',')
time_limit = args.time_limit
mths = args.mths.split(',')
start_id, rep, algo = args.start_id, args.rep, args.algo
save_dir = './data/eval_exps/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def evaluate_2rd_bandit(dataset, algo, time_limit, run_id, seed):
    print('HMAB-%s-%s: run_id=%d' % (dataset, algo, run_id))
    print('==> Start to Evaluate', dataset, 'Budget', time_limit)
    train_data, test_data = load_train_test_data(dataset)
    enable_intersect = True
    bandit = SecondLayerBandit(algo, train_data, per_run_time_limit=300,
                               seed=seed, eval_type='holdout',
                               mth='alter_hpo',
                               enable_intersection=enable_intersect)
    mth_id = 'hmab' if enable_intersect else 'hmab0'
    _start_time = time.time()
    _iter_id = 0
    stats = list()

    while True:
        if time.time() > time_limit + _start_time or bandit.early_stopped_flag:
            break
        res = bandit.play_once()
        print('Iteration %d - %.4f' % (_iter_id, res))
        stats.append([_iter_id, time.time() - _start_time, res])
        _iter_id += 1

    print(bandit.final_rewards)
    print(bandit.action_sequence)
    print(np.mean(bandit.evaluation_cost['fe']))
    print(np.mean(bandit.evaluation_cost['hpo']))

    fe_optimizer = bandit.optimizer['fe']
    final_train_data = fe_optimizer.apply(train_data, bandit.inc['fe'])
    assert final_train_data == bandit.inc['fe']
    final_test_data = fe_optimizer.apply(test_data, bandit.inc['fe'])
    config = bandit.inc['hpo']

    evaluator = ClassificationEvaluator(config, name='fe', seed=seed, resampling_strategy='holdout')
    val_score = evaluator(None, data_node=final_train_data)
    print('==> Best validation score', val_score, res)

    X_train, y_train = final_train_data.data
    clf = fetch_predict_estimator(config, X_train, y_train)
    X_test, y_test = final_test_data.data
    y_pred = clf.predict(X_test)
    test_score = balanced_accuracy(y_test, y_pred)
    print('==> Test score', test_score)

    # Alleviate overfitting.
    y_pred1 = bandit.predict(test_data.data[0])
    test_score1 = balanced_accuracy(y_test, y_pred1)
    print('==> Test score with average ensemble', test_score1)

    y_pred2 = bandit.predict(test_data.data[0], is_weighted=True)
    test_score2 = balanced_accuracy(y_test, y_pred2)
    print('==> Test score with weighted ensemble', test_score2)

    save_path = save_dir + '%s_2rd_bandit_%s_%d_%d_%s.pkl' % (mth_id, dataset, time_limit, run_id, algo)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, val_score, test_score, test_score1, test_score2], f)


def evaluate_ausk(dataset, algo, time_limit, run_id, seed):
    print('AUSK-%s-%s: run_id=%d' % (dataset, algo, run_id))
    print('==> Start to Evaluate', dataset, 'Budget', time_limit)
    include_models = [algo]
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_limit,
        include_preprocessors=None,
        n_jobs=1,
        include_estimators=include_models,
        ensemble_memory_limit=8192,
        ml_memory_limit=8192,
        ensemble_size=1,
        ensemble_nbest=1,
        initial_configurations_via_metalearning=0,
        per_run_time_limit=300,
        seed=int(seed),
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67}
    )
    print(automl)

    train_data, test_data = load_train_test_data(dataset)
    X, y = train_data.data
    feat_type = ['Categorical' if _type == CATEGORICAL else 'Numerical'
                 for _type in train_data.feature_types]

    from autosklearn.metrics import balanced_accuracy
    automl.fit(X.copy(), y.copy(), metric=balanced_accuracy, feat_type=feat_type)
    model_desc = automl.show_models()
    print(model_desc)
    val_result = np.max(automl.cv_results_['mean_test_score'])
    print('Best validation accuracy', val_result)

    X_test, y_test = test_data.data
    automl.refit(X.copy(), y.copy())
    y_pred = automl.predict(X_test)
    test_result = balanced_accuracy(y_test, y_pred)
    print('Test accuracy', test_result)

    save_path = save_dir + 'ausk_2rd_bandit_%s_%d_%d_%s.pkl' % (dataset, time_limit, run_id, algo)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, val_result, test_result, model_desc], f)


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
                        evaluate_ausk(dataset, algo, time_limit, run_id, seed)
                    else:
                        evaluate_2rd_bandit(dataset, algo, time_limit, run_id, seed)
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
                    task_id = '%s_2rd_bandit_%s_%d' % (mth, dataset, time_limit)
                    task_id += '_%d_%s.pkl' % (run_id, algo)
                    file_path = save_dir + task_id
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    val_acc, test_acc = data[1], data[2]
                    results.append([val_acc, test_acc])
                    if mth == 'ausk':
                        print(data)
                print(mth, results)
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
