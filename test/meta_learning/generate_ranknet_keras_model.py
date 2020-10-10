import os
import sys
sys.path.append(os.getcwd())
from solnml.components.meta_learning.algorithm_recomendation.ranknet_advisor import RankNetAdvisor
from solnml.components.utils.constants import MULTICLASS_CLS

if __name__ == "__main__":
    # refit ranknet on all datasets
    ranker = RankNetAdvisor(task_type=MULTICLASS_CLS, exclude_datasets=None, metric='bal_acc')
    # ranker = GBMAdvisor(task_type=MULTICLASS_CLS, exclude_datasets=test_datasets, metric='bal_acc')
    ranker.fit()