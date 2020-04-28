import os
import sys

sys.path.append(os.getcwd())
from automlToolkit.components.meta_learning.algorithm_recomendation.algorithm_advisor import AlgorithmAdvisor
from automlToolkit.components.meta_learning.algorithm_recomendation.meta_generator import get_feature_vector

alad = AlgorithmAdvisor().fit_meta_learner()
meta_feature = get_feature_vector('pc4')
algorithms, preds = alad.predict_meta_learner(meta_feature)
print(dict(zip(algorithms, preds)))
print(alad.fetch_algorithm_set('pc4'))
