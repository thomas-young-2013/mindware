import os
import sys

sys.path.append(os.getcwd())
from automlToolkit.components.meta_learning.algorithm_recomendation.algorithm_advisor import AlgorithmAdvisor
from automlToolkit.components.meta_learning.algorithm_recomendation.meta_generator import get_feature_vector
from automlToolkit.components.utils.constants import MULTICLASS_CLS

alad = AlgorithmAdvisor(task_type=MULTICLASS_CLS)
meta_infos = alad.fit_meta_learner()
datasets = [item[0] for item in meta_infos]
print(datasets)
test_datasets = ['pc4', 'page-blocks(1)', 'winequality_white', 'pollen']
for test_dataset in test_datasets:
    print(test_dataset in datasets)

    meta_feature = get_feature_vector(test_dataset, task_type=MULTICLASS_CLS)
    algorithms, preds = alad.predict_meta_learner(meta_feature)
    print(test_dataset)
    print(dict(zip(algorithms, preds)))
    print(alad.fetch_algorithm_set(test_dataset))
