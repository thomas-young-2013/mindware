import os
import sys

sys.path.append(os.getcwd())
from automlToolkit.components.meta_learning.algorithm_recomendation.algorithm_advisor import AlgorithmAdvisor
from automlToolkit.components.meta_learning.algorithm_recomendation.meta_generator import get_feature_vector
from automlToolkit.components.utils.constants import MULTICLASS_CLS

test_datasets = ['gina_prior2', 'pc2', 'abalone', 'wind', 'waveform-5000(2)', 'page-blocks(1)', 'winequality_white', 'pollen']
alad = AlgorithmAdvisor(task_type=MULTICLASS_CLS, exclude_datasets=test_datasets, n_algorithm=5)
meta_infos = alad.fit_meta_learner()
datasets = [item[0] for item in meta_infos]
print(datasets)
for test_dataset in test_datasets:
    print(test_dataset in datasets)
    meta_feature = get_feature_vector(test_dataset, task_type=MULTICLASS_CLS)
    algorithms, preds = alad.predict_meta_learner(meta_feature)

    print('='*50)
    print(test_dataset, test_dataset in datasets)
    from collections import OrderedDict
    print(alad.fetch_run_results(test_dataset))
    print(OrderedDict(zip(algorithms, preds)))
    print(alad.fetch_algorithm_set(test_dataset))
