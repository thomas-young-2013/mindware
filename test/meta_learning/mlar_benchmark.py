import os
import sys
import numpy as np
sys.path.append(os.getcwd())
from solnml.components.meta_learning.algorithm_recomendation.ranknet_advisor import RankNetAdvisor
from solnml.components.meta_learning.algorithm_recomendation.gbm_advisor import GBMAdvisor
from solnml.components.meta_learning.algorithm_recomendation.meta_generator import get_feature_vector
from solnml.components.utils.constants import MULTICLASS_CLS


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


"""
    Evaluation on 114 datasets:
    test_datasets = ['space_ga', 'yeast', 'fri_c0_1000_10', 'mfeat-fourier(1)', 'codrna', 'fri_c3_1000_5', 'pc4', 'fri_c3_1000_10', 'kc1', 'fri_c1_1000_5', 'fri_c2_1000_25', 'wind', 'spambase', 'fri_c0_1000_25', 'vehicle', 'pc3', 'electricity', 'adult-census', 'fri_c1_1000_10', 'ailerons', 'glass', 'musk', 'hypothyroid(1)', 'nursery', 'socmob', 'fri_c1_1000_25', 'delta_elevators', 'usps', 'colleges_usnews', 'magic_telescope', 'letter(1)', 'semeion', 'poker', 'credit', 'optdigits', 'mc1', 'mammography', 'waveform-5000(2)', 'kin8nm', 'sylva_prior', 'kropt', 'fri_c4_1000_10', '2dplanes', 'mfeat-fourier(2)', 'fri_c2_1000_10', 'fri_c3_1000_50', 'page-blocks(1)', 'house_16H', 'madelon', 'cpu_act', 'messidor_features', 'ionosphere', 'analcatdata_supreme', 'adult', 'car(1)', 'cmc', 'higgs', 'fri_c1_1000_50', 'winequality_white', 'winequality_red', 'satimage', 'gina_prior2', 'mushroom', 'car(2)', 'eeg', 'visualizing_soil', 'elevators', 'spectf', 'fri_c4_1000_50', 'segment', 'diabetes', 'covertype', 'amazon_employee', 'bank32nh', 'page-blocks(2)', 'houses', 'mv', 'puma32H', 'jm1', 'colleges_aaup', 'analcatdata_halloffame', 'pollen', 'fried', 'quake', 'fri_c2_1000_5', 'delta_ailerons', 'pc2', 'mfeat-morphological(1)', 'sick', 'lymphography', 'mfeat-factors(2)', 'isolet', 'letter(2)', 'a9a', 'fri_c2_1000_50', 'pendigits', 'dna', 'waveform-5000(1)', 'credit-g', 'fri_c0_1000_50', 'cal_housing', 'ap_omentum_ovary', 'mfeat-zernike(1)', 'fri_c0_1000_5', 'pc1', 'house_8L', 'mfeat-karhunen(2)', 'mfeat-morphological(2)', 'puma8NH', 'cpu_small', 'abalone', 'splice', 'balloon', 'rmftsa_sleepdata(2)']
"""
if __name__ == "__main__":
    np.random.seed(1)
    total_datasets = ['space_ga', 'yeast', 'fri_c0_1000_10', 'mfeat-fourier(1)', 'codrna', 'fri_c3_1000_5', 'pc4', 'fri_c3_1000_10', 'kc1', 'fri_c1_1000_5', 'fri_c2_1000_25', 'wind', 'spambase', 'fri_c0_1000_25', 'vehicle', 'pc3', 'electricity', 'adult-census', 'fri_c1_1000_10', 'ailerons', 'glass', 'musk', 'hypothyroid(1)', 'nursery', 'socmob', 'fri_c1_1000_25', 'delta_elevators', 'usps', 'colleges_usnews', 'magic_telescope', 'letter(1)', 'semeion', 'poker', 'credit', 'optdigits', 'mc1', 'mammography', 'waveform-5000(2)', 'kin8nm', 'sylva_prior', 'kropt', 'fri_c4_1000_10', '2dplanes', 'mfeat-fourier(2)', 'fri_c2_1000_10', 'fri_c3_1000_50', 'page-blocks(1)', 'house_16H', 'madelon', 'cpu_act', 'messidor_features', 'ionosphere', 'analcatdata_supreme', 'adult', 'car(1)', 'cmc', 'higgs', 'fri_c1_1000_50', 'winequality_white', 'winequality_red', 'satimage', 'gina_prior2', 'mushroom', 'car(2)', 'eeg', 'visualizing_soil', 'elevators', 'spectf', 'fri_c4_1000_50', 'segment', 'diabetes', 'covertype', 'amazon_employee', 'bank32nh', 'page-blocks(2)', 'houses', 'mv', 'puma32H', 'jm1', 'colleges_aaup', 'analcatdata_halloffame', 'pollen', 'fried', 'quake', 'fri_c2_1000_5', 'delta_ailerons', 'pc2', 'mfeat-morphological(1)', 'sick', 'lymphography', 'mfeat-factors(2)', 'isolet', 'letter(2)', 'a9a', 'fri_c2_1000_50', 'pendigits', 'dna', 'waveform-5000(1)', 'credit-g', 'fri_c0_1000_50', 'cal_housing', 'ap_omentum_ovary', 'mfeat-zernike(1)', 'fri_c0_1000_5', 'pc1', 'house_8L', 'mfeat-karhunen(2)', 'mfeat-morphological(2)', 'puma8NH', 'cpu_small', 'abalone', 'splice', 'balloon', 'rmftsa_sleepdata(2)']
    np.random.shuffle(total_datasets)
    n_fold = 10
    fold_size = len(total_datasets) // n_fold
    aps, top1 = list(), list()

    for i in range(n_fold):
        test_datasets = total_datasets[i * fold_size: (i+1) * fold_size]
        ranker = RankNetAdvisor(task_type=MULTICLASS_CLS, exclude_datasets=test_datasets, metric='bal_acc')
        # ranker = GBMAdvisor(task_type=MULTICLASS_CLS, exclude_datasets=test_datasets, metric='bal_acc')
        ranker.fit()

        for test_dataset in test_datasets:
            meta_feature = get_feature_vector(test_dataset, dataset_id=test_dataset, task_type=MULTICLASS_CLS)
            # preds = ranker.predict(meta_feature)
            # print(preds)
            pred_algos = ranker.fetch_algorithm_set(test_dataset, dataset_id=test_dataset)
            true_ranks = list(ranker.fetch_run_results(test_dataset).keys())
            ap = average_precision_atN(pred_algos[:5], true_ranks[:5])
            top1.append(1 if true_ranks[0] in pred_algos[:5] else 0)
            print('AP@5', ap)
            print('=' * 10)
            aps.append(ap)
        print('#Fold-%d' % i, np.mean(aps), 'Top1', np.mean(top1))
    print('Final AP@5', np.mean(aps), 'Top1', np.mean(top1))
