import os
import re
import sys
import argparse
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.components.fe_optimizers.bo_optimizer import BayesianOptimizationOptimizer
from solnml.components.utils.constants import CLASSIFICATION, MULTICLASS_CLS
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator
from solnml.components.models.classification import _classifiers
from solnml.utils.functions import get_increasing_sequence
from solnml.components.transfer_learning.tlbo.tlbo_taff_optimizer import TLBO_AF


test_datasets = ['splice', 'segment', 'abalone', 'delta_ailerons', 'space_ga',
                 'pollen', 'quake', 'wind', 'dna', 'spambase', 'satimage',
                 'waveform-5000(1)', 'optdigits', 'madelon', 'kr-vs-kp', 'isolet',
                 'analcatdata_supreme', 'balloon', 'waveform-5000(2)', 'gina_prior2']

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='fe')
parser.add_argument('--algo', type=str, default='random_forest')
parser.add_argument('--metric', type=str, default='bal_acc')
parser.add_argument('--mths', type=str, default='tlbo_taff')
parser.add_argument('--plot_mode', type=int, default=0)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--max_runs', type=int, default=50)
parser.add_argument('--datasets', type=str, default=','.join(test_datasets))

args = parser.parse_args()

data_dir = 'data/tlbo/'
algo_name = args.algo
metric = args.metric

benchmark = args.benchmark
task_id = benchmark
hist_dir = 'test/bayesian_opt/runhistory/%s_%s/' % (benchmark, algo_name)

rep = args.rep
start_id = args.start_id
max_runs = args.max_runs
mths = args.mths.split(',')
datasets = args.datasets.split(',')
plot_mode = args.plot_mode
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def get_datasets():
    _datasets = list()
    pattern = r'(.*)-%s-%s-%s.pkl' % (algo_name, metric, task_id)
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
        _filename = '%s-%s-%s-%s.pkl' % (dataset, algo_name, metric, task_id)
        with open(hist_dir + _filename, 'rb') as f:
            data = pk.load(f)
        runhistory.append((metafeature_dict[dataset], list(data.items())))
    return runhistory


def get_configspace():
    if benchmark == 'hpo':
        cs = _classifiers[algo_name].get_hyperparameter_search_space()
        model = UnParametrizedHyperparameter("estimator", algo_name)
        cs.add_hyperparameter(model)
        return cs

    train_data, test_data = load_train_test_data('splice', task_type=MULTICLASS_CLS)
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


def evaluate(mode, dataset, run_id, metric):
    print(mode, dataset, run_id, metric)

    metric = get_metric(metric)
    train_data, test_data = load_train_test_data(dataset, task_type=MULTICLASS_CLS)

    cs = _classifiers[algo_name].get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", algo_name)
    cs.add_hyperparameter(model)
    default_hpo_config = cs.get_default_configuration()

    fe_evaluator = ClassificationEvaluator(default_hpo_config, scorer=metric,
                                           name='fe', resampling_strategy='holdout',
                                           seed=1)

    hpo_evaluator = ClassificationEvaluator(default_hpo_config, scorer=metric,
                                            data_node=train_data, name='hpo',
                                            resampling_strategy='holdout',
                                            seed=1)

    fe_optimizer = BayesianOptimizationOptimizer(task_type=CLASSIFICATION,
                                                 input_data=train_data,
                                                 evaluator=fe_evaluator,
                                                 model_id=algo_name,
                                                 time_limit_per_trans=600,
                                                 mem_limit_per_trans=5120,
                                                 number_of_unit_resource=10,
                                                 seed=1)

    def objective_function(config):
        if benchmark == 'fe':
            return fe_optimizer.evaluate_function(config)
        else:
            return hpo_evaluator(config)

    meta_feature_vec = metafeature_dict[dataset]
    past_datasets = test_datasets.copy()
    if dataset in past_datasets:
        past_datasets.remove(dataset)
    past_history = load_runhistory(past_datasets)

    tlbo = TLBO_AF(objective_function, config_space, past_history,
                   dataset_metafeature=meta_feature_vec,
                   max_runs=max_runs,
                   acq_method='taff2')

    tlbo.run()
    print('TLBO result')
    print(tlbo.get_incumbent())
    runs = [tlbo.configurations, tlbo.perfs]
    perf = tlbo.history_container.incumbent_value

    file_saved = '%s_%s_result_%d_%d_%s.pkl' % (mode, dataset, max_runs, run_id, benchmark)
    with open(data_dir + file_saved, 'wb') as f:
        pk.dump([perf, runs], f)


for dataset in datasets:
    if plot_mode != 1:
        for mth in mths:
            for run_id in range(start_id, start_id + rep):
                evaluate(mth, dataset, run_id, metric)
    else:
        print('='*10)
        # cmp_methods = ['tlbo_gpoe', 'tlbo_no-unct', 'tlbo_indp-aspt', 'lite_bo']
        cmp_methods = mths
        perfs = list()
        for mth in cmp_methods:
            _result = list()
            _runs = list()
            for run_id in range(start_id, start_id + rep):
                file_saved = '%s_%s_result_%d_%d_%s.pkl' % (mth, dataset, max_runs, run_id, benchmark)

                with open(data_dir + file_saved, 'rb') as f:
                    perf, runs = pk.load(f)
                _result.append(perf)
                inc_seq = get_increasing_sequence(-np.array(runs[1]))
                while len(inc_seq) < max_runs:
                    inc_seq.append(inc_seq[-1])
                _runs.append(inc_seq)

            perfs.append(np.mean(np.array(_runs), axis=0))
            print(dataset.ljust(20), mth.ljust(10), '%.4f\u00B1%.4f' % (np.mean(_result), np.std(_result)))
        # print(perfs)

        fig = plt.figure()
        ax = plt.subplot(111)
        x = np.arange(1, max_runs+1)
        y_min, y_max = 1, 0
        for idx, y in enumerate(perfs):
            ax.plot(x, y, label=cmp_methods[idx])
            if y[0] >= 0:
                y_min = min(y_min, y[0])
            y_max = max(y_max, y[-1])
        plt.title(dataset)
        ax.legend()
        epsilon = (y_max - y_min) * 0.05
        plt.ylim(y_min - epsilon, y_max + epsilon)
        plt.show()
