import os
import sys
import time
import pickle
import argparse
import numpy as np
from tabulate import tabulate
sys.path.append(os.getcwd())
from automlToolkit.components.evaluators.evaluator import Evaluator, fetch_predict_estimator
from automlToolkit.components.metrics.cls_metrics import balanced_accuracy
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.datasets.utils import load_train_test_data

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,vehicle,yeast,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--mode', type=str, choices=['alter', 'rb', 'alter_p', 'alter_hpo', 'plot'], default='rb')
parser.add_argument('--cv', type=str, choices=['cv', 'holdout'], default='holdout')
parser.add_argument('--algo', type=str, default='random_forest')
parser.add_argument('--time_cost', type=int, default=600)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep_num', type=int, default=10)

project_dir = './'
save_folder = project_dir + 'data/debug_2rdmab/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def evaluate_2rd_layered_bandit(run_id, mth='rb', dataset='pc4', algo='libsvm_svc',
                                cv='holdout', time_limit=120000, seed=1):
    train_data, test_data = load_train_test_data(dataset)
    bandit = SecondLayerBandit(algo, train_data, dataset_id=dataset, mth=mth, seed=seed, eval_type=cv)

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

    evaluator = Evaluator(config, name='fe', seed=seed, resampling_strategy='holdout')
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

    save_path = save_folder + '%s_%s_%d_%d_%s.pkl' % (mth, dataset, time_limit, run_id, algo)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, val_score, test_score, test_score1, test_score2], f)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo = args.algo
    time_cost = args.time_cost
    mode = args.mode
    cv = args.cv
    np.random.seed(1)
    rep = args.rep_num
    start_id = args.start_id
    seeds = np.random.randint(low=1, high=10000, size=start_id+rep)
    dataset_list = dataset_str.split(',')

    if mode != 'plot':
        for dataset in dataset_list:
            for _id in range(start_id, start_id+rep):
                seed = seeds[_id]
                print('Running %s with %d-th seed' % (dataset, _id + 1))
                if mode == 'alter':
                    evaluate_2rd_layered_bandit(_id, mth=mode, dataset=dataset, cv=cv,
                                                algo=algo, time_limit=time_cost, seed=seed)
                elif mode == 'rb':
                    evaluate_2rd_layered_bandit(_id, mth=mode, dataset=dataset, algo=algo, cv=cv,
                                                time_limit=time_cost, seed=seed)
                elif mode in ['alter_p', 'alter_hpo']:
                    evaluate_2rd_layered_bandit(_id, mth=mode, dataset=dataset, algo=algo, cv=cv,
                                                time_limit=time_cost, seed=seed)
                else:
                    raise ValueError('Invalid mode: %s!' % mode)
    else:
        headers = ['dataset']
        method_ids = ['alter', 'alter_hpo']
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
                    val_acc, test_acc = data[3], data[4]
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
