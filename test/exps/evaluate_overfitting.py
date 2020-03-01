import os
import sys
import argparse
import pickle
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer
from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.evaluators.evaluator import Evaluator, fetch_predict_estimator

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='credit')
parser.add_argument('--iter_num', type=int, default=100)
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)

save_dir = './data/exp_results/overfit/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def conduct_hpo(dataset='pc4', classifier_id='random_forest', iter_num=100, run_id=0, seed=1):
    from autosklearn.pipeline.components.classification import _classifiers

    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)

    raw_data, test_raw_data = load_train_test_data(dataset, random_state=seed)
    evaluator = Evaluator(cs.get_default_configuration(), name='hpo', data_node=raw_data,
                          resampling_strategy='cv', seed=seed)

    optimizer = SMACOptimizer(
        evaluator, cs, trials_per_iter=2,
        output_dir='logs', per_run_time_limit=180
    )
    task_id = 'hpo-%s-%s-%d' % (dataset, classifier_id, iter_num)
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    for _iter in range(iter_num):
        perf, _, config = optimizer.iterate()
        val_acc_list.append(perf)
        estimator = fetch_predict_estimator(config, raw_data.data[0], raw_data.data[1])
        pred = estimator.predict(raw_data.data[0])
        train_perf = accuracy_score(raw_data.data[1], pred)
        train_acc_list.append(train_perf)
        pred = estimator.predict(test_raw_data.data[0])
        test_perf = accuracy_score(test_raw_data.data[1], pred)
        test_acc_list.append(test_perf)
        print(train_acc_list)
        print(val_acc_list)
        print(test_acc_list)

    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([train_acc_list, val_acc_list, test_acc_list], f)

    train_iter_list, val_iter_list = [], []
    configs = optimizer.configs
    perfs = optimizer.perfs
    for i, config in enumerate(configs):
        val_iter_list.append(perfs[i])
        estimator = fetch_predict_estimator(config, raw_data.data[0], raw_data.data[1])
        pred = estimator.predict(raw_data.data[0])
        train_perf = accuracy_score(raw_data.data[1], pred)
        train_iter_list.append(train_perf)

    save_path = save_dir + 'iter-%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([train_iter_list, val_iter_list], f)


def conduct_fe(dataset='pc4', classifier_id='random_forest', iter_num=100, run_id=0, seed=1):
    from autosklearn.pipeline.components.classification import _classifiers

    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)
    default_config = cs.get_default_configuration()

    raw_data, test_raw_data = load_train_test_data(dataset, random_state=seed)
    evaluator = Evaluator(default_config, name='fe', data_node=raw_data,
                          resampling_strategy='cv', seed=seed)

    optimizer = EvaluationBasedOptimizer(task_type='classification', input_data=raw_data, evaluator=evaluator,
                                         model_id=classifier_id,
                                         time_limit_per_trans=240, mem_limit_per_trans=10000, seed=seed)

    task_id = 'fe-%s-%s-%d' % (dataset, classifier_id, iter_num)
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    for _iter in range(iter_num):
        perf, _, incubent = optimizer.iterate()
        val_acc_list.append(perf)
        train_node = optimizer.apply(raw_data, incubent)
        test_node = optimizer.apply(test_raw_data, incubent)
        estimator = fetch_predict_estimator(default_config, train_node.data[0], train_node.data[1])
        pred = estimator.predict(train_node.data[0])
        train_perf = accuracy_score(train_node.data[1], pred)
        train_acc_list.append(train_perf)
        pred = estimator.predict(test_node.data[0])
        test_perf = accuracy_score(test_node.data[1], pred)
        test_acc_list.append(test_perf)
        print(train_acc_list)
        print(val_acc_list)
        print(test_acc_list)

    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([train_acc_list, val_acc_list, test_acc_list], f)


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == '__main__':
    args = parser.parse_args()
    datasets = args.datasets
    iter_num = args.iter_num
    rep = args.rep_num

    dataset_list = datasets.split(',')
    check_datasets(dataset_list)

    mode_list = ['hpo', 'fe']
    algo_list = ['random_forest', 'xgradient_boosting', 'libsvm_svc', 'k_nearest_neighbors']
    for dataset in dataset_list:
        for run_id in range(rep):
            for algo in algo_list:
                for mode in mode_list:
                    if mode == 'hpo':
                        conduct_hpo(dataset=dataset, classifier_id=algo, iter_num=iter_num, run_id=run_id)
                    elif mode == 'fe':
                        conduct_fe(dataset=dataset, classifier_id=algo, iter_num=iter_num, run_id=run_id)
