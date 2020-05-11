import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from automlToolkit.components.meta_learning.algorithm_recomendation.algorithm_advisor import AlgorithmAdvisor
from automlToolkit.components.meta_learning.algorithm_recomendation.meta_generator import get_feature_vector
from automlToolkit.components.utils.constants import MULTICLASS_CLS

test_datasets = ['gina_prior2', 'pc2', 'abalone', 'wind', 'waveform-5000(2)', 'page-blocks(1)', 'winequality_white', 'pollen']
alad = AlgorithmAdvisor(task_type=MULTICLASS_CLS, exclude_datasets=test_datasets, n_algorithm=5, metric='bal_acc')
meta_infos = alad.fit_meta_learner()
datasets = [item[0] for item in meta_infos]
print(datasets)


# 1.0, 2.0
def topk(l1, l2):
    score = 0
    for item in l1[:3]:
        if item in l2[:5]:
            score += 1
    return score


scores = list()
for test_dataset in test_datasets:
    print(test_dataset in datasets)
    meta_feature = get_feature_vector(test_dataset, dataset_id=test_dataset, task_type=MULTICLASS_CLS)
    algorithms, preds = alad.predict_meta_learner(meta_feature)

    print('='*50)
    print(test_dataset, test_dataset in datasets)
    runs = alad.fetch_run_results(test_dataset)
    print(runs)
    list_true = list(runs.keys())
    list_pred = alad.fetch_algorithm_set(test_dataset)
    print(list_true[:5])
    print(list_pred)
    scores.append(topk(list_true, list_pred))

print('average score', np.mean(scores))
