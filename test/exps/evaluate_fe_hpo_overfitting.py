import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cmc')
parser.add_argument('--algo', type=str, default='k_nearest_neighbors,libsvm_svc,random_forest,xgradient_boosting')
data_dir = '/Users/thomasyoung/Desktop/overfitting/overfit/'


def get_avg(dataset, algo, arm_type):
    val_result, test_result = list(), list()
    for _id in range(5):
        data_path = data_dir + '%s-%s-%s-100-%d.pkl' % (arm_type, dataset, algo, _id)
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                val_result.append(data[0])
                test_result.append(data[1])
    val_result, test_result = np.mean(val_result, axis=0), np.mean(test_result, axis=0)
    return val_result, test_result


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    algos = args.algo.split(',')
    fig = plt.figure(figsize=(10, 6))

    for i in range(len(algos)):
        algo = algos[i]
        plt.subplot(2, 2, i+1)
        val_fe, test_fe = get_avg(dataset, algo, 'fe')
        val_hpo, test_hpo = get_avg(dataset, algo, 'hpo')
        print(val_fe)
        print(test_fe)
        print(val_hpo)
        print(test_hpo)
        x = np.linspace(1, 100, 100, endpoint=True)
        plt.plot(x, val_fe, color="red", label='fe_val', linestyle='--')
        plt.plot(x, test_fe, color="red", label='fe_test')

        plt.plot(x, val_hpo, color="blue", label='hpo_val', linestyle='--')
        plt.plot(x, test_hpo, color="blue", label='hpo_test')
        plt.title('%s - %s' % (dataset, algo))
        plt.legend(loc='upper left', frameon=False)

    plt.show()
