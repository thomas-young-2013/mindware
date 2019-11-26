import os
import sys
import pickle
import argparse
import numpy as np
from tabulate import tabulate
import autosklearn.classification

proj_dir = '/home/thomas/PycharmProjects/Feature-Engineering/'
if not os.path.exists(proj_dir):
    proj_dir = './'
sys.path.append(proj_dir)
from evaluate_transgraph import engineer_data
from evaluator import Evaluator
from utils.default_random_forest import DefaultRandomForest

parser = argparse.ArgumentParser()
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--time_limit', type=int, default=120)
parser.add_argument('--n_job', type=int, default=1)
parser.add_argument('--mode', type=int, default=0)
dataset_list = 'credit,diabetes,pc4,sick,spectf,splice,waveform,' \
               'winequality_red,winequality_white,ionosphere,amazon_employee,' \
               'lymphography,messidor_features,spambase,ap_omentum_ovary,a9a'
dataset_list2 = 'eeg,higgs,kropt,madelon,mushroom,quake,satimage,semeion'
dataset_list3 = 'messidor_features,lymphography,winequality_red,winequality_white,credit,' \
                'ionosphere,splice,diabetes,pc4,spectf,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default='pc4')
args = parser.parse_args()


def extract_data(data):
    results = list()
    for item in data:
        result = list()
        result.append(item[0])
        for res in item[1:]:
            result.append(res[0])
        results.append(result)
    return results


if args.datasets == 'all':
    data_dir = proj_dir + 'data/datasets'
    datasets = [item.split('.')[0] for item in os.listdir(data_dir) if item.endswith('csv')]
else:
    datasets = args.datasets.split(',')
time_limit = args.time_limit


def evaluate_ausk_fe(dataset, time_limit, fe_mth='none', ratio=0.5, ensb_size=1, include_models=None, seed=1):
    print('==> Start to evaluate', dataset, 'budget', time_limit)
    n_job = args.n_job
    # includes_fe = ['no_preprocessing'] if fe is not 'ausk' else None
    includes_fe = None if fe_mth == 'none' else ['no_preprocessing']

    # Construct the ML model.
    def get_automl(seed, time_budget):
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time_budget,
            include_preprocessors=includes_fe,
            n_jobs=n_job,
            include_estimators=include_models,
            ensemble_memory_limit=8192,
            ml_memory_limit=8192,
            ensemble_size=ensb_size,
            ensemble_nbest=ensb_size,
            initial_configurations_via_metalearning=0,
            seed=seed,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5}
        )
        print(automl)
        return automl

    data_node, fe_time = engineer_data(dataset, fe_mth, time_budget=int(ratio*time_limit), seed=seed)
    budget_left = int(time_limit - fe_time)
    print('Available budget for automl', budget_left)

    X, y = data_node.data
    automl = get_automl(seed, budget_left)
    automl.fit(X.copy(), y.copy())
    model_desc = automl.show_models()
    print(model_desc)

    all_test_results = automl.cv_results_['mean_test_score']
    print('Mean test score', all_test_results)
    best_result = np.max(automl.cv_results_['mean_test_score'])
    print('Validation Accuracy', best_result)

    return [best_result, all_test_results, model_desc]


def evaluate_fe(dataset, time_limit, fe='eval_base', seed=1):
    np.random.seed(seed)
    train_data, _ = engineer_data(dataset, fe, time_budget=time_limit)

    cs = DefaultRandomForest.get_hyperparameter_search_space()
    config = cs.get_default_configuration().get_dictionary()
    clf = DefaultRandomForest(**config)
    evaluator = Evaluator(seed=seed, clf=clf)
    score = evaluator(train_data)
    print('==> Validation score', score)

    raw_data, _ = engineer_data(dataset, 'none', time_budget=time_limit)
    base_score = evaluator(raw_data)
    print('==> Base validation score', base_score)
    return [score, [base_score], []]


def evaluate_fe_compoment():
    """
    Evaluate the features constructed by Auto-SKlearn and Evaluation-based method.
    :return:
    """
    # Add random forest classifier (with default hyperparameter) component to auto-sklearn.
    from utils.default_random_forest import DefaultRandomForest
    autosklearn.pipeline.components.classification.add_classifier(DefaultRandomForest)
    # cs = DefaultRandomForest.get_hyperparameter_search_space()
    # print(cs)

    headers = ['AUSK', 'EB', 'ER']
    save_template = proj_dir + 'data/ausk_cv_result_exp1_%s_%d.pkl'
    include_models = ['DefaultRandomForest']
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
            res2 = evaluate_fe(dataset, time_limit, fe='eval_base', seed=seed)
            # res1 = evaluate_ausk_fe(dataset, time_limit, include_models=include_models, seed=seed)
            # res3 = evaluate_fe(dataset, time_limit, fe='epd_rdc', seed=seed)
            exp_data.append([dataset, res2])
            with open(save_path, 'wb') as f:
                pickle.dump(exp_data, f)

        data = extract_data(exp_data)
        print(tabulate(data, headers, tablefmt="github", floatfmt=".4f"))


def evalaute_ausk_with_fe():
    """
    Evaluate the baselines: 1) only HPO, 2) FE+HPO, 3) Our method.
    :return:
    """
    headers = ['NONE', 'AUSK', 'OURS']
    include_models = ['random_forest']

    seed = 1
    mode = args.mode
    save_template = proj_dir + 'data/ausk_cv_result_exp2_%d_%s_%d.pkl'
    for dataset in datasets:
        exp_data = list()
        save_path = save_template % (mode, dataset, time_limit)
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                exp_data = pickle.load(f)
        else:
            if mode == 0:
                res1 = evaluate_ausk_fe(dataset, time_limit, include_models=include_models, seed=seed)
            else:
                res1 = evaluate_ausk_fe(dataset, time_limit, fe_mth='eval_base', ratio=0.5,
                                        include_models=include_models, seed=seed)
            exp_data.append([dataset, res1])
            with open(save_path, 'wb') as f:
                pickle.dump(exp_data, f)
        data = extract_data(exp_data)
        print(tabulate(data, headers, tablefmt="github", floatfmt=".4f"))


def evaluate_base_result():
    seed = 1
    data_dir = proj_dir + 'data/datasets'
    datasets = [item.split('.')[0] for item in os.listdir(data_dir) if item.endswith('csv')]
    results = list()
    save_path = proj_dir + 'data/ausk_cv_result_exp3.pkl'
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
    else:
        for dataset in datasets:
            data_node, _ = engineer_data(dataset, 'none', time_budget=time_limit, seed=seed)
            cs = DefaultRandomForest.get_hyperparameter_search_space()
            config = cs.get_default_configuration().get_dictionary()
            clf = DefaultRandomForest(**config)
            evaluator = Evaluator(seed=seed, clf=clf)
            score = evaluator(data_node)
            print('==> Validation score', score)
            results.append([dataset, score])
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
    print(results)


if __name__ == "__main__":
    evaluate_fe_compoment()
    # evalaute_ausk_with_fe()
    # evaluate_base_result()
