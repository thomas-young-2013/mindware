import os
import re
import sys
import argparse
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

test_datasets = ['splice', 'segment', 'abalone', 'delta_ailerons', 'space_ga',
                 'pollen', 'quake', 'wind', 'dna', 'spambase', 'satimage',
                 'waveform-5000(1)', 'optdigits', 'madelon', 'kr-vs-kp', 'isolet',
                 'analcatdata_supreme', 'balloon', 'waveform-5000(2)', 'gina_prior2']

parser = argparse.ArgumentParser()
parser.add_argument('--mth', type=str, default='tlbo')
parser.add_argument('--plot_mode', type=int, default=0)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--max_runs', type=int, default=50)
parser.add_argument('--datasets', type=str, default=','.join(test_datasets))

args = parser.parse_args()

data_dir = 'data/tlbo/'
hist_dir = 'test/bayesian_opt/runhistory/config_res/'
task_id = 'fe'
algo_name = 'random_forest'
metric = 'acc'
rep = args.rep
start_id = args.start_id
max_runs = args.max_runs
mode = args.mth
datasets = args.datasets.split(',')
plot_mode = args.plot_mode
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def get_datasets():
    _datasets = list()
    pattern = r'(.*)-%s-%s-%d-%s.pkl' % (algo_name, metric, 0, task_id)
    for filename in os.listdir(hist_dir):
        result = re.search(pattern, filename, re.M | re.I)
        if result is not None:
            _datasets.append(result.group(1))
    print(_datasets)
    return _datasets


print(len(datasets))


def get_metafeature_vector(metafeature_dict):
    sorted_keys = sorted(metafeature_dict.keys())
    return np.array([metafeature_dict[key] for key in sorted_keys])


with open(hist_dir + '../metafeature.pkl', 'rb') as f:
    metafeature_dict = pk.load(f)
    for dataset in metafeature_dict.keys():
        vec = get_metafeature_vector(metafeature_dict[dataset])
        metafeature_dict[dataset] = vec


def load_runhistory(dataset_names):
    runhistory = list()
    for dataset in dataset_names:
        _filename = '%s-%s-%s-%d-%s.pkl' % (dataset, 'random_forest', 'acc', 0, task_id)
        with open(hist_dir + _filename, 'rb') as f:
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
    train_data, test_data = load_train_test_data('splice')
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
if mode.startswith('tlbo') and plot_mode != 1:
    gp_models_dict = pretrain_gp_models(config_space)


def evaluate(dataset, run_id, metric):
    print(dataset, run_id, metric)

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
    if mode == 'bo':
        bo = BO(objective_function, hyper_space, max_runs=max_runs)
        bo.run()
        print('BO result')
        print(bo.get_incumbent())
        perf = bo.history_container.incumbent_value
        runs = [bo.configurations, bo.perfs]
    elif mode == 'lite_bo':
        from litebo.facade.bo_facade import BayesianOptimization
        bo = BayesianOptimization(objective_function, hyper_space, max_runs=max_runs)
        bo.run()
        print('BO result')
        print(bo.get_incumbent())
        perf = bo.history_container.incumbent_value
        runs = [bo.configurations, bo.perfs]
    elif mode.startswith('tlbo'):
        _, gp_fusion = mode.split('_')
        meta_feature_vec = metafeature_dict[dataset]
        past_datasets = test_datasets.copy()
        if dataset in past_datasets:
            past_datasets.remove(dataset)
        past_history = load_runhistory(past_datasets)

        gp_models = [gp_models_dict[dataset_name] for dataset_name in past_datasets]
        tlbo = TLBO(objective_function, hyper_space, past_history, gp_models=gp_models,
                    dataset_metafeature=meta_feature_vec, max_runs=max_runs, gp_fusion=gp_fusion)
        tlbo.run()
        print('TLBO result')
        print(tlbo.get_incumbent())
        runs = [tlbo.configurations, tlbo.perfs]
        perf = tlbo.history_container.incumbent_value
    else:
        raise ValueError('Invalid mode.')
    with open(data_dir + '%s_%s_result_%d_%d.pkl' % (mode, dataset, max_runs, run_id), 'wb') as f:
        pk.dump([perf, runs], f)


for dataset in datasets:
    if plot_mode != 1:
        for run_id in range(start_id, start_id + rep):
            evaluate(dataset, run_id, metric)
    else:
        with open(data_dir + '%s_result_%d_%d_%s.pkl' % (mode, max_runs, rep, dataset), 'rb') as f:
            data = pk.load(f)
        data = [item[0] for item in data]
        mu, std = np.mean(data), np.std(data)
        print(data[0], '%.4f\u00B1%.4f' % (mu, std))
