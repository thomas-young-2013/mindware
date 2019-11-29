import os
import sys
import pickle
import argparse
import numpy as np
from tabulate import tabulate

proj_dir = '/home/thomas/PycharmProjects/feature-engieering-toolkit/'
if not os.path.exists(proj_dir):
    proj_dir = './'
sys.path.append(proj_dir)
from evaluate_transgraph import engineer_data
from utils.default_random_forest import DefaultRandomForest
from evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=1200)
dataset_list = 'messidor_features,lymphography,winequality_red,winequality_white,credit,' \
                'ionosphere,splice,diabetes,pc4,spectf,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_list)
args = parser.parse_args()


if args.datasets == 'all':
    data_dir = proj_dir + 'data/datasets'
    datasets = [item.split('.')[0] for item in os.listdir(data_dir) if item.endswith('csv')]
else:
    datasets = args.datasets.split(',')
time_limit = args.time_limit


def evaluate_fe(dataset, time_limit, fe='eval_base', seed=1):
    np.random.seed(seed)
    from sklearn.neighbors import KNeighborsClassifier
    surrogate_evaluator = Evaluator(seed=seed, clf=KNeighborsClassifier())

    train_data, _ = engineer_data(dataset, fe, evaluator=surrogate_evaluator, time_budget=time_limit, seed=seed)
    print('Best surrogate score', train_data.score)

    cs = DefaultRandomForest.get_hyperparameter_search_space()
    config = cs.get_default_configuration().get_dictionary()
    clf = DefaultRandomForest(**config, random_state=seed)
    evaluator = Evaluator(seed=seed, clf=clf)

    score = evaluator(train_data)
    print('==> Validation score', score)

    return [train_data.score, score]


def evaluate_surrogate():
    headers = ['Dataset', 'Surrogate', 'Target']
    save_template = proj_dir + 'data/fe_cv_result_exp1_%s_%d.pkl'
    seed = 1

    for dataset in datasets:
        exp_data = list()
        save_path = save_template % (dataset, time_limit)
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                exp_data = pickle.load(f)
                print(exp_data)
        else:
            # Test the performance of AutoSklearn.
            res1 = evaluate_fe(dataset, time_limit, fe='eval_base', seed=seed)
            exp_data.append([dataset, res1[0], res1[1]])
            with open(save_path, 'wb') as f:
                pickle.dump(exp_data, f)
        data = exp_data
        print(tabulate(data, headers, tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    evaluate_surrogate()
