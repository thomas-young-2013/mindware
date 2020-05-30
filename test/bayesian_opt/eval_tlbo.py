import os
import re
import sys
import pickle as pk
import numpy as np

sys.path.append(os.getcwd())

from solnml.components.transfer_learning.tlbo.tlbo_optimizer import TLBO
from solnml.components.transfer_learning.tlbo.bo_optimizer import BO
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.components.fe_optimizers.bo_optimizer import BayesianOptimizationOptimizer
from solnml.components.utils.constants import CLASSIFICATION, REGRESSION
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator
from solnml.components.models.classification import _classifiers
from solnml.components.transfer_learning.tlbo.models.gp_ensemble import create_gp_model
from solnml.components.transfer_learning.tlbo.config_space.util import convert_configurations_to_array

task_id = 'fe'
algo_name = 'random_forest'
metric = 'acc'
datasets = list()

pattern = r'(.*)-%s-%s-%d-%s.pkl' % (algo_name, metric, 0, task_id)
data_dir = 'test/bayesian_opt/runhistory/config_res/'
for filename in os.listdir(data_dir):
    result = re.search(pattern, filename, re.M | re.I)
    if result is not None:
        datasets.append(result.group(1))
print(datasets)

test_datasets = ['cpu_act', 'mfeat-morphological(2)', 'poker', 'mfeat-zernike(1)',
                 'pendigits', 'hypothyroid(1)', 'winequality_red', 'delta_ailerons', 'colleges_usnews',
                 'page-blocks(1)', 'sick', 'pc2', 'analcatdata_halloffame', 'nursery',
                 'credit-g', 'puma32H', 'mammography', 'electricity', 'abalone', 'fried',
                 'satimage', 'fri_c1_1000_25', 'puma8NH']


def get_metafeature_vector(metafeature_dict):
    sorted_keys = sorted(metafeature_dict.keys())
    return np.array([metafeature_dict[key] for key in sorted_keys])


with open(data_dir + '../metafeature.pkl', 'rb') as f:
    metafeature_dict = pk.load(f)
    for dataset in metafeature_dict.keys():
        vec = get_metafeature_vector(metafeature_dict[dataset])
        metafeature_dict[dataset] = vec


def load_runhistory(dataset_names):
    runhistory = list()
    for dataset in dataset_names:
        _filename = '%s-%s-%s-%d-%s.pkl' % (dataset, 'random_forest', 'acc', 0, task_id)
        with open(data_dir + _filename, 'rb') as f:
            data = pk.load(f)
        runhistory.append((metafeature_dict[dataset], list(data.items())))
    return runhistory


def pretrain_gp_models(config_space):
    runhistory = load_runhistory(test_datasets)
    gp_models = dict()
    for dataset, hist in zip(test_datasets, runhistory):
        gp_model = create_gp_model(config_space)
        X = list()
        for row in hist[1]:
            conf_vector = convert_configurations_to_array([row[0]])[0]
            X.append(conf_vector)
        X = np.array(X)
        y = np.array([row[1] for row in hist[1]]).reshape(-1, 1)

        gp_model.train(X, y)
        gp_models[dataset] = gp_model
        print('%s: training basic GP model finished.' % dataset)
    return gp_models


def get_configspace():
    train_data, test_data = load_train_test_data('pc2')
    cs = _classifiers[algo_name].get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", algo_name)
    cs.add_hyperparameter(model)
    default_hpo_config = cs.get_default_configuration()
    fe_evaluator = ClassificationEvaluator(default_hpo_config, scorer=metric,
                                           name='fe', resampling_strategy='holdout',
                                           seed=1)
    fe_optimizer = BayesianOptimizationOptimizer(task_type=CLASSIFICATION,
                                                 input_data=train_data,
                                                 evaluator=fe_evaluator,
                                                 model_id=algo_name,
                                                 time_limit_per_trans=600,
                                                 mem_limit_per_trans=5120,
                                                 number_of_unit_resource=10,
                                                 seed=1)
    hyper_space = fe_optimizer.hyperparameter_space
    return hyper_space


eval_result = list()
config_space = get_configspace()
gp_models_dict = pretrain_gp_models(config_space)


def evaluate(dataset, run_id, metric):
    """
        One vs Rest evaluation.
    """
    print(dataset, run_id, metric)
    meta_feature_vec = metafeature_dict[dataset]
    past_datasets = test_datasets.copy()
    past_datasets.remove(dataset)
    past_history = load_runhistory(past_datasets)

    metric = get_metric(metric)
    train_data, test_data = load_train_test_data(dataset)

    cs = _classifiers[algo_name].get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", algo_name)
    cs.add_hyperparameter(model)
    default_hpo_config = cs.get_default_configuration()
    fe_evaluator = ClassificationEvaluator(default_hpo_config, scorer=metric,
                                           name='fe', resampling_strategy='holdout',
                                           seed=1)
    fe_optimizer = BayesianOptimizationOptimizer(task_type=CLASSIFICATION,
                                                 input_data=train_data,
                                                 evaluator=fe_evaluator,
                                                 model_id=algo_name,
                                                 time_limit_per_trans=600,
                                                 mem_limit_per_trans=5120,
                                                 number_of_unit_resource=10,
                                                 seed=1)
    hyper_space = fe_optimizer.hyperparameter_space

    def objective_function(config):
        return fe_optimizer.evaluate_function(config)
    mode = 'tlbo'
    if mode == 'bo':
        bo = BO(objective_function, hyper_space, max_runs=20)
        bo.run()
        print('BO result')
        print(bo.get_incumbent())
        perf_bo = bo.history_container.incumbent_value
        perf_tlbo = 1.0
    elif mode == 'tlbo':
        perf_bo = 1.0
        gp_models = [gp_models_dict[dataset_name] for dataset_name in past_datasets]
        tlbo = TLBO(objective_function, hyper_space, past_history, gp_models=gp_models, dataset_metafeature=meta_feature_vec, max_runs=20)
        tlbo.run()
        print('TLBO result')
        print(tlbo.get_incumbent())
        perf_tlbo = tlbo.history_container.incumbent_value
    else:
        raise ValueError('Invalid mode.')
    return perf_bo, perf_tlbo


for dataset in test_datasets[:5]:
    rep = 5
    result = list()
    for run_id in range(rep):
        perf_bo, perf_tlbo = evaluate(dataset, run_id, metric)
        result.append((perf_bo, perf_tlbo))
    mean_res = np.mean(result, axis=0)
    print(dataset, mean_res)
    eval_result.append((dataset, mean_res))

print(eval_result)


"""
[('cpu_act',                array([-0.93820311, -0.93712436])), 
 ('mfeat-morphological(2)', array([-0.99810606, -0.99810606])), 
 ('poker',                  array([-0.68409091, -0.68017677])), 
 ('mfeat-zernike(1)',       array([-0.79103535, -0.78219697])), 
 ('pendigits',              array([-0.99425683, -0.99172984])), 
 ('hypothyroid(1)',         array([-0.99531459, -0.99531459])), 
 ('winequality_red',        array([-0.6572104 , -0.65248227])), 
 ('delta_ailerons',         array([-0.94066596, -0.94155154])), -
 ('colleges_usnews',        array([-0.77325581, -0.77325581])), 
 ('page-blocks(1)',         array([-0.97531719, -0.97739331])), -
 ('sick',                   array([-0.97824632, -0.9832664 ])), -
 ('pc2',                    array([-0.99593496, -0.99593496])), 
 ('analcatdata_halloffame', array([-0.97457627, -0.96986817])), 
 ('nursery',                array([-0.98353789, -0.98246639])), 
 ('credit-g',               array([-0.78409091, -0.79166667])), -
 ('puma32H',                array([-0.88750193, -0.89613192])), -
 ('mammography',            array([-0.98848629, -0.98747037])), 
 ('electricity',            array([-0.89222324, -0.89016133])), 
 ('abalone',                array([-0.63251738, -0.63674826])), -
 ('fried',                  array([-0.91727833, -0.91486265])), 
 ('satimage',               array([-0.90439733, -0.90714566])), -
 ('fri_c1_1000_25',         array([-0.90782828, -0.9229798 ])), -
 ('puma8NH',                array([-0.82632147, -0.8247804 ]))]


rep=5
[('cpu_act', array([ 1.        , -0.93804901])), 
 ('mfeat-morphological(2)', array([ 1.        , -0.99810606])), 
 ('poker', array([ 1.        , -0.67484848])), 
 ('mfeat-zernike(1)', array([ 1.        , -0.79015152])), 
 ('pendigits', array([ 1.        , -0.99228119]))]
# new tlbo
[('cpu_act', array([ 1.        , -0.93712436])), 
 ('mfeat-morphological(2)', array([ 1.        , -0.99810606])), 
 ('poker', array([ 1.        , -0.66348485])), 
 ('mfeat-zernike(1)', array([ 1.        , -0.78219697])), 
 ('pendigits', array([ 1.        , -0.99172984]))]

[('cpu_act', array([-0.93758669,  1.        ])), 
 ('mfeat-morphological(2)', array([-0.99810606,  1.        ])), 
 ('poker', array([-0.67757576,  1.        ])), 
 ('mfeat-zernike(1)', array([-0.78636364,  1.        ])), 
 ('pendigits', array([-0.99228119,  1.        ]))]

# new bo
[('cpu_act', array([-0.93749422,  1.        ])), 
 ('mfeat-morphological(2)', array([-0.99810606,  1.        ])), 
 ('poker', array([-0.67772727,  1.        ])), 
 ('mfeat-zernike(1)', array([-0.78257576,  1.        ])), 
 ('pendigits', array([-0.99172984,  1.        ]))]

"""
