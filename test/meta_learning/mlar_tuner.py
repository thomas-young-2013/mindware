import os
import sys
import numpy as np
from litebo.config_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from litebo.facade.bo_facade import BayesianOptimization

sys.path.append(os.getcwd())
from solnml.components.meta_learning.algorithm_recomendation.ranknet_advisor import RankNetAdvisor
from solnml.components.meta_learning.algorithm_recomendation.gbm_advisor import GBMAdvisor
from solnml.components.meta_learning.algorithm_recomendation.metadata_manager import get_feature_vector
from solnml.components.utils.constants import MULTICLASS_CLS


total_datasets = ['space_ga', 'yeast', 'fri_c0_1000_10', 'mfeat-fourier(1)', 'codrna', 'fri_c3_1000_5', 'pc4',
                  'fri_c3_1000_10', 'kc1', 'fri_c1_1000_5', 'fri_c2_1000_25', 'wind', 'spambase', 'fri_c0_1000_25',
                  'vehicle', 'pc3', 'electricity', 'adult-census', 'fri_c1_1000_10', 'ailerons', 'glass', 'musk',
                  'hypothyroid(1)', 'nursery', 'socmob', 'fri_c1_1000_25', 'delta_elevators', 'usps', 'colleges_usnews',
                  'magic_telescope', 'letter(1)', 'semeion', 'poker', 'credit', 'optdigits', 'mc1', 'mammography',
                  'waveform-5000(2)', 'kin8nm', 'sylva_prior', 'kropt', 'fri_c4_1000_10', '2dplanes',
                  'mfeat-fourier(2)', 'fri_c2_1000_10', 'fri_c3_1000_50', 'page-blocks(1)', 'house_16H', 'madelon',
                  'cpu_act', 'messidor_features', 'ionosphere', 'analcatdata_supreme', 'adult', 'car(1)', 'cmc',
                  'higgs', 'fri_c1_1000_50', 'winequality_white', 'winequality_red', 'satimage', 'gina_prior2',
                  'mushroom', 'car(2)', 'eeg', 'visualizing_soil', 'elevators', 'spectf', 'fri_c4_1000_50', 'segment',
                  'diabetes', 'covertype', 'amazon_employee', 'bank32nh', 'page-blocks(2)', 'houses', 'mv', 'puma32H',
                  'jm1', 'colleges_aaup', 'analcatdata_halloffame', 'pollen', 'fried', 'quake', 'fri_c2_1000_5',
                  'delta_ailerons', 'pc2', 'mfeat-morphological(1)', 'sick', 'lymphography', 'mfeat-factors(2)',
                  'isolet', 'letter(2)', 'a9a', 'fri_c2_1000_50', 'pendigits', 'dna', 'waveform-5000(1)', 'credit-g',
                  'fri_c0_1000_50', 'cal_housing', 'ap_omentum_ovary', 'mfeat-zernike(1)', 'fri_c0_1000_5', 'pc1',
                  'house_8L', 'mfeat-karhunen(2)', 'mfeat-morphological(2)', 'puma8NH', 'cpu_small', 'abalone',
                  'splice', 'balloon', 'rmftsa_sleepdata(2)']
np.random.shuffle(total_datasets)


def build_configspace():
    cs = ConfigurationSpace()
    l1_size = UniformIntegerHyperparameter("layer1_size", 16, 128, default_value=64)
    l2_size = UniformIntegerHyperparameter("layer2_size", 16, 128, default_value=32)
    activation = CategoricalHyperparameter("activation", choices=['relu', 'tanh'], default_value='relu')
    batch_size = UniformIntegerHyperparameter("batch_size", 16, 256, default_value=64)
    cs.add_hyperparameters([l1_size, l2_size, activation, batch_size])
    return cs


def obj_function(config):
    config = config.get_dictionary()

    n_fold = 5
    fold_size = len(total_datasets) // n_fold
    aps, top1 = list(), list()

    for i in range(n_fold):
        test_datasets = total_datasets[i * fold_size: (i+1) * fold_size]
        ranker = RankNetAdvisor(task_type=MULTICLASS_CLS, exclude_datasets=test_datasets, metric='bal_acc')
        # ranker = GBMAdvisor(task_type=MULTICLASS_CLS, exclude_datasets=test_datasets, metric='bal_acc')
        ranker.fit(**config)

        for test_dataset in test_datasets:
            pred_algos = ranker.fetch_algorithm_set(test_dataset, dataset_id=test_dataset)
            true_ranks = list(ranker.fetch_run_results(test_dataset).keys())
            ap = average_precision_atN(pred_algos[:5], true_ranks[:5])
            top1.append(1 if true_ranks[0] in pred_algos[:5] else 0)
            print('AP@5', ap)
            print('=' * 10)
            aps.append(ap)
    return -np.mean(aps)


def average_precision_atN(preds, true_labels):
    N = len(preds)
    precision_ = list()
    for i in range(1, N+1):
        if preds[i-1] in true_labels:
            _pre = (len(precision_)+1)/i
            precision_.append(_pre)
    if len(precision_) == 0:
        return 0
    return np.sum(precision_) / N


def tune_meta_learner():
    cs = build_configspace()
    def_value = obj_function(cs.get_default_configuration())
    print("Default Value: %.2f" % (def_value))

    bo = BayesianOptimization(obj_function, cs, max_runs=50, time_limit_per_trial=1200)
    bo.run()
    inc_value = bo.get_incumbent()
    config = inc_value[0][0]

    print('Best hyperparameter config found', config)
    return config


if __name__ == "__main__":
    tune_meta_learner()
