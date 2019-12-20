import os
import sys
import time
import pickle
import tabulate
import argparse
import numpy as np
sys.path.append(os.getcwd())
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.datasets.utils import load_data

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--mode', type=str, choices=['alter', 'rb', 'alter-rb', 'plot', 'all'], default='both')
parser.add_argument('--cv', type=str, choices=['cv', 'holdout'], default='holdout')
parser.add_argument('--algo', type=str, default='random_forest')
parser.add_argument('--time_cost', type=int, default=10800)
parser.add_argument('--iter_num', type=int, default=100)
parser.add_argument('--rep_num', type=int, default=5)

project_dir = './'


def evaluate_2rd_layered_bandit(run_id, mth='rb', dataset='pc4',
                                algo='libsvm_svc', cv='holdout',
                                iter_num=100,
                                time_limit=120000,
                                seed=1):
    raw_data = load_data(dataset, datanode_returned=True)
    strategy = 'avg' if mth != 'alter-rb' else 'rb'
    mth_id = mth if mth != 'alter-rb' else 'alter'
    bandit = SecondLayerBandit(algo, raw_data, dataset_id=dataset, mth=mth_id,
                               strategy=strategy, seed=seed, eval_type=cv)

    _start_time = time.time()
    stats = list()

    for _iter in range(iter_num):
        _iter_start_time = time.time()
        bandit.play_once()
        stats.append([iter, time.time() - _start_time])

        if time.time() > time_limit + _start_time:
            break

        print('%s%s' % ('\n', '='*65))
        end_time = time.time()
        print('| %s-%s-%d | Iteration-%d: %.4f | Time_cost: %.2f-%.2f |' %
              (dataset, mth, run_id, _iter, bandit.final_rewards[-1],
               end_time - _iter_start_time, end_time - _start_time))
        print('='*65, '\n')

        # Save the intermediate result.
        save_folder = project_dir + 'data/2rdlayer-mab/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_path = save_folder + '%s-%d_2rdlayer-mab_%s_%s_%d_%d_%s.pkl' % (
            mth, seed, dataset, algo, iter_num, time_cost, cv)
        data = [bandit.final_rewards, bandit.action_sequence, bandit.evaluation_cost, stats]
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    algo = args.algo
    iter_num = args.iter_num
    time_cost = args.time_cost
    mode = args.mode
    cv = args.cv
    np.random.seed(1)
    seeds = np.random.randint(low=1, high=10000, size=args.rep_num)

    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set
    else:
        dataset_list = dataset_str.split(',')

    if mode != 'plot':
        for dataset in dataset_list:
            for _id, seed in enumerate(seeds):
                print('Running %s with %d-th seed' % (dataset, _id + 1))
                if mode == 'alter':
                    evaluate_2rd_layered_bandit(_id, mth=mode, dataset=dataset, cv=cv,
                                                algo=algo, iter_num=iter_num, time_limit=time_cost, seed=seed)
                elif mode == 'rb':
                    evaluate_2rd_layered_bandit(_id, mth=mode, dataset=dataset, algo=algo, cv=cv,
                                                iter_num=iter_num, time_limit=time_cost, seed=seed)
                elif mode == 'alter-rb':
                    evaluate_2rd_layered_bandit(_id, mth=mode, dataset=dataset, algo=algo, cv=cv,
                                                iter_num=iter_num, time_limit=time_cost, seed=seed)
                elif mode == 'all':
                    evaluate_2rd_layered_bandit(_id, mth='alter', dataset=dataset, algo=algo, cv=cv,
                                                iter_num=iter_num, time_limit=time_cost, seed=seed)
                    evaluate_2rd_layered_bandit(_id, mth='rb', dataset=dataset, algo=algo, cv=cv,
                                                iter_num=iter_num, time_limit=time_cost, seed=seed)
                    evaluate_2rd_layered_bandit(_id, mth='alter-rb', dataset=dataset, algo=algo, cv=cv,
                                                iter_num=iter_num, time_limit=time_cost, seed=seed)
                else:
                    raise ValueError('Invalid mode: %s!' % mode)
    else:
        headers = ['dataset', 'rb-mean', 'rb-var', 'alter-mean', 'alter-var', 'alter-rb-mean', 'alter-rb-var']
        tbl_data = list()
        for dataset in dataset_list:
            row_data = [dataset]
            for mth in ['rb', 'alter', 'alter-rb']:
                results = list()
                for seed in seeds:
                    save_folder = project_dir + 'data/2rdlayer-mab/'
                    file_path = save_folder + '%s-%d_2rdlayer-mab_%s_%s_%d_%d_%s.pkl' % (
                        mth, seed, dataset, algo, iter_num, time_cost, cv)
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    final_rewards, action_sequence, evaluation_cost, _ = data
                    results.append(final_rewards)
                if len(results) == len(seeds):
                    for item in results:
                        cur_num = len(item)
                        if cur_num < 1 + iter_num:
                            item.extend([item[-1]] * (1+iter_num - cur_num))
                        assert len(item) == iter_num + 1

                    mean_values = np.mean(results, axis=0)
                    std_value = np.std(np.asarray(results)[:, -1])
                    row_data.append('%.2f%%' % (100*mean_values[-1]))
                    row_data.append('%.4f' % std_value)
                    print('='*30)
                    print('%s-%s: %.2f%%' % (dataset, mth, 100*mean_values[-1]))
                    print('-'*30)
                    print(mean_values)
                    print('='*30)
                else:
                    row_data.extend(['-', '-'])

            tbl_data.append(row_data)
        print(tabulate.tabulate(tbl_data, headers, tablefmt='github'))
