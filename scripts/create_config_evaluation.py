import os
import sys
import pickle
import argparse
import numpy as np

sys.path.append(os.getcwd())
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator
from solnml.components.models.classification import _classifiers
from solnml.components.utils.constants import MULTICLASS_CLS, BINARY_CLS, REGRESSION, CLS_TASKS
from solnml.datasets.utils import load_data
from solnml.components.metrics.metric import get_metric

parser = argparse.ArgumentParser()
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--datasets', type=str, default='diabetes')
parser.add_argument('--metric', type=str, default='bal_acc')
parser.add_argument('--algo', type=str, default='random_forest')
args = parser.parse_args()

datasets = args.datasets.split(',')
start_id, rep = args.start_id, args.rep
save_dir = './data/config_res/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def evaluate_ml_algorithm(dataset, algo, run_id, obj_metric, seed=1, task_type=None):
    print('EVALUATE-%s-%s-%s: run_id=%d' % (dataset, algo, obj_metric, run_id))
    train_data = load_data(dataset, task_type=task_type, datanode_returned=True)
    print(set(train_data.data[1]))
    metric = get_metric(obj_metric)

    cs = _classifiers[algo].get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", algo)
    cs.add_hyperparameter(model)
    default_hpo_config = cs.get_default_configuration()
    hpo_evaluator = ClassificationEvaluator(default_hpo_config, scorer=metric,
                                            data_node=train_data, name='hpo',
                                            resampling_strategy='holdout',
                                            seed=seed)
    hpo_optimizer = SMACOptimizer(evaluator=hpo_evaluator,
                                  config_space=cs,
                                  per_run_time_limit=600,
                                  per_run_mem_limit=5120,
                                  output_dir='./logs',
                                  trials_per_iter=10)
    hpo_optimizer.iterate()
    hpo_eval_dict = dict()
    for key, value in hpo_optimizer.eval_dict.items():
        hpo_eval_dict[key[1]] = value

    save_path = save_dir + '%s-%s-%s-%d-hpo.pkl' % (dataset, algo, obj_metric, run_id)
    with open(save_path, 'wb') as f:
        pickle.dump(hpo_eval_dict, f)


def check_datasets(datasets, task_type=None):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, task_type=task_type)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == "__main__":
    algo = args.algo
    task_type = MULTICLASS_CLS
    metric = args.metric

    check_datasets(datasets, task_type=task_type)
    running_info = list()
    log_filename = 'running-%d.txt' % os.getpid()

    for dataset in datasets:
        np.random.seed(1)
        seeds = np.random.randint(low=1, high=10000, size=start_id + rep)

        for run_id in range(start_id, start_id + rep):
            seed = seeds[run_id]
            try:
                task_id = '%s-%s-%s-%d: %s' % (dataset, algo, metric, run_id, 'success')
                evaluate_ml_algorithm(dataset, algo, run_id, metric,
                                      seed=seed, task_type=task_type)
            except Exception as e:
                task_id = '%s-%s-%s-%d: %s' % (dataset, algo, metric, run_id, str(e))

            print(task_id)
            running_info.append(task_id)
            with open(save_dir + log_filename, 'a') as f:
                f.write('\n' + task_id)

    # Write down the error info.
    with open(save_dir + 'failed-%s' % log_filename, 'w') as f:
        f.write('\n'.join(running_info))
