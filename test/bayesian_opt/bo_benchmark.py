import os
import sys
import time
import numpy as np
import pickle as pk

sys.path.append(os.getcwd())

from solnml.components.transfer_learning.tlbo.bo_optimizer import BO
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.models.classification import _classifiers


test_datasets = ['cpu_act', 'mfeat-morphological(2)', 'poker', 'mfeat-zernike(1)',
                 'pendigits', 'hypothyroid(1)', 'winequality_red', 'delta_ailerons', 'colleges_usnews',
                 'page-blocks(1)', 'sick', 'pc2', 'analcatdata_halloffame', 'nursery',
                 'credit-g', 'puma32H', 'mammography', 'electricity', 'abalone', 'fried',
                 'satimage', 'fri_c1_1000_25', 'puma8NH']

print(len(test_datasets))

algo_name = 'lightgbm'
max_runs = 70
rep = 5


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
    train_data, test_data = load_train_test_data(dataset, test_size=0.3)

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


eval_result = list()

for dataset in test_datasets:
    for mth in ['gp_bo', 'lite_bo', 'smac']:
        result = list()
        for run_id in range(rep):
            perf_bo = evaluate(mth, dataset, run_id)
            result.append(perf_bo)
        mean_res = np.mean(result)
        print(dataset, mth, mean_res)
        eval_result.append((dataset, mth, mean_res))

print(eval_result)
with open('data/bo_benchmar_data_%d.pkl' % time.time(), 'wb') as f:
    pk.dump(eval_result, f)
