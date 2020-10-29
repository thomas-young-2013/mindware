import os
import sys
import argparse
import pickle
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

sys.path.append(os.getcwd())
from solnml.components.optimizers.smac_optimizer import SMACOptimizer
from solnml.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer
from solnml.datasets.utils import load_train_test_data
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator, fetch_predict_estimator
from solnml.components.metrics.cls_metrics import balanced_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='credit')
parser.add_argument('--algos', type=str, default='random_forest')
parser.add_argument('--iter_num', type=int, default=100)
parser.add_argument('--rep_num', type=int, default=10)
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
    evaluator = ClassificationEvaluator(cs.get_default_configuration(), name='hpo', data_node=raw_data,
                                        resampling_strategy='holdout', seed=seed)

    default_config = cs.get_default_configuration()
    val_acc = 1. - evaluator(default_config)
    estimator = fetch_predict_estimator(default_config, raw_data.data[0], raw_data.data[1])
    pred = estimator.predict(test_raw_data.data[0])
    test_acc = balanced_accuracy(test_raw_data.data[1], pred)

    optimizer = SMACOptimizer(
        evaluator, cs, trials_per_iter=2,
        output_dir='logs', per_run_time_limit=180
    )
    task_id = 'hpo-%s-%s-%d' % (dataset, classifier_id, iter_num)

    val_acc_list, test_acc_list = [], []
    val_acc_list.append(val_acc)
    test_acc_list.append(test_acc)

    for _iter in range(iter_num):
        perf, _, config = optimizer.iterate()
        val_acc_list.append(perf)
        estimator = fetch_predict_estimator(config, raw_data.data[0], raw_data.data[1])
        pred = estimator.predict(test_raw_data.data[0])
        test_perf = balanced_accuracy(test_raw_data.data[1], pred)
        test_acc_list.append(test_perf)
        print(val_acc_list)
        print(test_acc_list)

    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([val_acc_list, test_acc_list], f)


def conduct_fe(dataset='pc4', classifier_id='random_forest', iter_num=100, run_id=0, seed=1):
    from autosklearn.pipeline.components.classification import _classifiers

    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)
    default_config = cs.get_default_configuration()

    raw_data, test_raw_data = load_train_test_data(dataset, random_state=seed)
    evaluator = ClassificationEvaluator(default_config, name='fe', data_node=raw_data,
                                        resampling_strategy='holdout', seed=seed)

    val_acc = evaluator(default_config)
    estimator = fetch_predict_estimator(default_config, raw_data.data[0], raw_data.data[1])
    pred = estimator.predict(test_raw_data.data[0])
    test_acc = balanced_accuracy(test_raw_data.data[1], pred)

    optimizer = EvaluationBasedOptimizer(task_type='classification', input_data=raw_data, evaluator=evaluator, model_id=classifier_id,
                                         time_limit_per_trans=240, mem_limit_per_trans=10000, seed=seed)

    task_id = 'fe-%s-%s-%d' % (dataset, classifier_id, iter_num)
    val_acc_list, test_acc_list = [], []

    val_acc_list.append(val_acc)
    test_acc_list.append(test_acc)

    for _iter in range(iter_num):
        perf, _, incubent = optimizer.iterate()
        val_acc_list.append(perf)
        train_node = optimizer.apply(raw_data, incubent)
        test_node = optimizer.apply(test_raw_data, incubent)
        estimator = fetch_predict_estimator(default_config, train_node.data[0], train_node.data[1])
        pred = estimator.predict(test_node.data[0])
        test_perf = balanced_accuracy(test_node.data[1], pred)
        test_acc_list.append(test_perf)
        print(val_acc_list)
        print(test_acc_list)

    save_path = save_dir + '%s-%d.pkl' % (task_id, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump([val_acc_list, test_acc_list], f)


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
    algo_list = args.algos.split(',')
    for dataset in dataset_list:
        for run_id in range(rep):
            for algo in algo_list:
                for mode in mode_list:
                    if mode == 'hpo':
                        conduct_hpo(dataset=dataset, classifier_id=algo, iter_num=iter_num, run_id=run_id)
                    elif mode == 'fe':
                        conduct_fe(dataset=dataset, classifier_id=algo, iter_num=iter_num, run_id=run_id)
