import os
import sys
import numpy as np
sys.path.append(os.getcwd())
from solnml.components.meta_learning.algorithm_recomendation.ranknet_advisor import RankNetAdvisor
from solnml.components.meta_learning.algorithm_recomendation.gbm_advisor import GBMAdvisor
from solnml.components.meta_learning.algorithm_recomendation.metadata_manager import get_feature_vector
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


if __name__ == "__main__":
    np.random.seed(1)
    total_datasets = ['abalone', 'adult', 'adult-census', 'ailerons', 'amazon_employee', 'analcatdata_halloffame', 'analcatdata_supreme', 'balloon', 'bank32nh', 'baseball', 'cal_housing', 'car(1)', 'car(2)', 'cmc', 'colleges_aaup', 'colleges_usnews', 'covertype', 'cpu_act', 'cpu_small', 'credit', 'credit-g', 'delta_ailerons', 'delta_elevators', 'diabetes', 'dna', 'eeg', 'elevators', 'fri_c0_1000_10', 'fri_c0_1000_25', 'fri_c0_1000_5', 'fri_c0_1000_50', 'fri_c1_1000_10', 'fri_c1_1000_25', 'fri_c1_1000_5', 'fri_c1_1000_50', 'fri_c2_1000_25', 'fri_c2_1000_5', 'fri_c2_1000_50', 'fri_c3_1000_10', 'fri_c3_1000_25', 'fri_c3_1000_5', 'fri_c3_1000_50', 'fri_c4_1000_10', 'fri_c4_1000_25', 'fri_c4_1000_50', 'gina_prior2', 'glass', 'house_16H', 'houses', 'hypothyroid(1)', 'hypothyroid(2)', 'ionosphere', 'isolet', 'jm1', 'kc1', 'kr-vs-kp', 'kropt', 'letter(1)', 'letter(2)', 'lymphography', 'madelon', 'magic_telescope', 'mammography', 'messidor_features', 'mfeat-factors(1)', 'mfeat-factors(2)', 'mfeat-fourier(1)', 'mfeat-fourier(2)', 'mfeat-karhunen(1)', 'mfeat-karhunen(2)', 'mfeat-morphological(1)', 'mfeat-morphological(2)', 'mfeat-zernike(1)', 'mfeat-zernike(2)', 'mnist_784', 'mushroom', 'musk', 'mv', 'nursery', 'optdigits', 'page-blocks(1)', 'pc1', 'pc3', 'pc4', 'pendigits', 'poker', 'pol', 'pollen', 'puma32H', 'puma8NH', 'quake', 'rmftsa_sleepdata(1)', 'rmftsa_sleepdata(2)', 'satimage', 'semeion', 'sick', 'socmob', 'space_ga', 'spectf', 'splice', 'sylva_prior', 'usps', 'vehicle', 'vehicle_sensIT', 'waveform-5000(1)', 'waveform-5000(2)', 'wind', 'winequality_red', 'yeast']
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
            meta_feature = get_feature_vector(test_dataset)
            pred_algos = ranker.fetch_algorithm_set(test_dataset)
            true_ranks = list(ranker.fetch_run_results(test_dataset).keys())
            print(pred_algos)
            print(true_ranks)
            ap = average_precision_atN(pred_algos[:5], true_ranks[:5])
            top1.append(1 if true_ranks[0] in pred_algos[:5] else 0)
            print('AP@5', ap)
            print('=' * 10)
            aps.append(ap)
        print('#Fold-%d' % i, np.mean(aps), 'Top1', np.mean(top1))
    print('Final AP@5', np.mean(aps), 'Top1', np.mean(top1))
