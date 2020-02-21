from typing import Callable
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Standard regression scores
r2 = make_scorer('r2', r2_score)
mean_squared_error = make_scorer('mean_squared_error',
                                 mean_squared_error,
                                 optimum=0,
                                 greater_is_better=False)
mean_absolute_error = make_scorer('mean_absolute_error',
                                  mean_absolute_error,
                                  optimum=0,
                                  greater_is_better=False)
median_absolute_error = make_scorer('median_absolute_error',
                                    median_absolute_error,
                                    optimum=0,
                                    greater_is_better=False)

# Standard Classification Scores
accuracy = make_scorer('accuracy', accuracy_score)
f1 = make_scorer('f1', f1_score)

# Score functions that need decision values
roc_auc = make_scorer('roc_auc',
                      roc_auc_score,
                      greater_is_better=True,
                      needs_threshold=True)
precision = make_scorer('precision', precision_score)
recall = make_scorer('recall', recall_score)


REGRESSION_METRICS = dict()
for scorer in [r2, mean_squared_error, mean_absolute_error,
               median_absolute_error]:
    REGRESSION_METRICS[scorer.name] = scorer

CLASSIFICATION_METRICS = dict()

for scorer in [accuracy, roc_auc, precision, recall, f1]:
    CLASSIFICATION_METRICS[scorer.name] = scorer


def fetch_scorer(_scorer: [str, Callable], task_type: str):
    if isinstance(scorer, str):
        if task_type == 'classification':
            return CLASSIFICATION_METRICS[_scorer]
        else:
            return REGRESSION_METRICS[_scorer]
    else:
        return _scorer
