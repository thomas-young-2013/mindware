import os
import sys
import time
import argparse
import numpy as np
import pickle as pk

sys.path.append(os.getcwd())

from solnml.components.transfer_learning.tlbo.bo_optimizer import BO
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.models.classification import _classifiers
from solnml.components.utils.constants import MULTICLASS_CLS

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='libsvm_svc')
parser.add_argument('--datasets', type=str, default='splice')

args = parser.parse_args()
test_datasets = args.datasets.split(',')
print(len(test_datasets))
algo_name = args.algo
max_runs = 70
rep = 10


def get_configspace():
    cs = _classifiers[algo_name].get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", algo_name)
    cs.add_hyperparameter(model)
    return cs


def get_estimator(config):
    from solnml.components.models.classification import _classifiers, _addons
    classifier_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    config_['random_state'] = 1
    try:
        estimator = _classifiers[classifier_type](**config_)
    except:
        estimator = _addons.components[classifier_type](**config_)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', 2)
    return classifier_type, estimator


def evaluate(mth, dataset, run_id):
    print(mth, dataset, run_id)
    train_data, test_data = load_train_test_data(dataset, test_size=0.3, task_type=MULTICLASS_CLS)

    def objective_function(config):
        metric = get_metric('bal_acc')
        _, estimator = get_estimator(config.get_dictionary())
        X_train, y_train = train_data.data
        X_test, y_test = test_data.data
        estimator.fit(X_train, y_train)
        return -metric(estimator, X_test, y_test)

    config_space = get_configspace()

    if mth == 'gp_bo':
        bo = BO(objective_function, config_space, max_runs=max_runs)
        bo.run()
        print('new BO result')
        print(bo.get_incumbent())
        perf_bo = bo.history_container.incumbent_value
    elif mth == 'lite_bo':
        from litebo.facade.bo_facade import BayesianOptimization
        bo = BayesianOptimization(objective_function, config_space, max_runs=max_runs)
        bo.run()
        print('lite BO result')
        print(bo.get_incumbent())
        perf_bo = bo.history_container.incumbent_value
    elif mth == 'smac':
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_facade import SMAC
        # Scenario object
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": max_runs,
                             "cs": config_space,
                             "deterministic": "true"
                             })
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=objective_function)
        incumbent = smac.optimize()
        perf_bo = objective_function(incumbent)
        print('SMAC BO result')
        print(perf_bo)
    else:
        raise ValueError('Invalid method.')
    return perf_bo


def check_datasets(datasets, task_type=MULTICLASS_CLS):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1, task_type=task_type)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


check_datasets(test_datasets)
for dataset in test_datasets:
    for mth in ['gp_bo', 'lite_bo', 'smac']:
        result = list()
        for run_id in range(rep):
            perf_bo = evaluate(mth, dataset, run_id)
            result.append(perf_bo)
        mean_res = np.mean(result)
        std_res = np.std(result)
        print(dataset, mth, mean_res, std_res)
        with open('data/bo_benchmark_%s_%s_%s.pkl' % (mth, algo_name, dataset), 'wb') as f:
            pk.dump((dataset, mth, mean_res, std_res), f)
