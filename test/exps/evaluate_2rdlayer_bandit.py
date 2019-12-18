import os
import sys
import time
import pickle
import argparse
import numpy as np
sys.path.append(os.getcwd())
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.datasets.utils import load_data

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--mode', type=str, choices=['alter', 'rb', 'alter-rb', 'plot', 'both'], default='both')
parser.add_argument('--algo', type=str, default='random_forest')
parser.add_argument('--time_cost', type=int, default=3600)
parser.add_argument('--iter_num', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)

project_dir = './'


def evaluate_2rd_layered_bandit(mth='rb', dataset='pc4',
                                algo='libsvm_svc',
                                iter_num=100,
                                time_limit=120000,
                                seed=1, strategy='avg'):
    raw_data = load_data(dataset, datanode_returned=True)
    bandit = SecondLayerBandit(algo, raw_data, mth=mth, strategy=strategy, seed=seed)

    _start_time = time.time()
    stats = list()

    for _iter in range(iter_num):
        res = bandit.play_once()
        stats.append([iter, time.time() - _start_time, res])

        if time.time() > time_limit + _start_time:
            break
        print('Iteration-%d: %.4f' % (_iter, bandit.final_rewards[-1]))

    if strategy == 'avg':
        save_path = project_dir + 'data/%s_2rdlayer_mab_%s_%s_%d_%d.pkl' % (mth, dataset, algo, iter_num, time_cost)
    else:
        save_path = project_dir + 'data/%s_2rdlayer_mab_%s_%s_%d_%d_%s.pkl' % (
            mth, dataset, algo, iter_num, time_cost, strategy)
    data = [bandit.final_rewards, bandit.action_sequence, bandit.evaluation_cost]
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(bandit.final_rewards)
    print(bandit.action_sequence)
    print(bandit.evaluation_cost['fe'])
    print(bandit.evaluation_cost['hpo'])
    if np.all(np.asarray(bandit.evaluation_cost['fe']) != None):
        print(np.mean(bandit.evaluation_cost['fe']))
    if np.all(np.asarray(bandit.evaluation_cost['hpo']) != None):
        print(np.mean(bandit.evaluation_cost['hpo']))


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo = args.algo
    iter_num = args.iter_num
    time_cost = args.time_cost
    mode = args.mode
    seed = args.seed

    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set
    else:
        dataset_list = dataset_str.split(',')

    for dataset in dataset_list:
        if mode == 'alter':
            evaluate_2rd_layered_bandit(mth='alter', dataset=dataset,
                                        algo=algo, iter_num=iter_num, time_limit=time_cost, seed=seed)
        elif mode == 'rb':
            evaluate_2rd_layered_bandit(mth='rb', dataset=dataset, algo=algo,
                                        iter_num=iter_num, time_limit=time_cost, seed=seed)
        elif mode == 'alter-rb':
            evaluate_2rd_layered_bandit(mth='alter', dataset=dataset, algo=algo,
                                        iter_num=iter_num, strategy='rb', time_limit=time_cost, seed=seed)
        elif mode == 'both':
            evaluate_2rd_layered_bandit(mth='alter', dataset=dataset, algo=algo,
                                        iter_num=iter_num, time_limit=time_cost, seed=seed)
            evaluate_2rd_layered_bandit(mth='rb', dataset=dataset, algo=algo,
                                        iter_num=iter_num, time_limit=time_cost, seed=seed)
        else:
            for mth in ['rb', 'alter', 'alter-rb']:
                if mth in ['alter', 'rb']:
                    save_path = project_dir + 'data/%s_2rdlayer_mab_%s_%s_%d_%d.pkl' % \
                                (mth, dataset, algo, iter_num, time_cost)
                else:
                    strategy = 'rb'
                    save_path = project_dir + 'data/%s_2rdlayer_mab_%s_%s_%d_%d_%s.pkl' % (
                        'alter', dataset, algo, iter_num, time_cost, strategy)
                if not os.path.exists(save_path):
                    continue
                with open(save_path, 'rb') as f:
                    data = pickle.load(f)
                final_rewards, action_sequence, evaluation_cost = data
                print('='*30, dataset, mth)
                print('='*50)
                print(final_rewards)
                print(action_sequence)
                print(evaluation_cost['fe'])
                print(evaluation_cost['hpo'])
                if np.all(np.asarray(evaluation_cost['fe']) != None):
                    print(np.mean(evaluation_cost['fe']))
                if np.all(np.asarray(evaluation_cost['hpo']) != None):
                    print(np.mean(evaluation_cost['hpo']))
                print('='*50)
