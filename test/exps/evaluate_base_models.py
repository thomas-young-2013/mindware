import os
import sys
import time
import pickle
import argparse
import numpy as np
import autosklearn.classification
from tabulate import tabulate

sys.path.append(os.getcwd())

from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.metrics.cls_metrics import balanced_accuracy
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter


def evaluate_base_model(classifier_id, dataset):
    _start_time = time.time()
    train_data, test_data = load_train_test_data(dataset)

    from autosklearn.pipeline.components.classification import _classifiers
    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)
    default_config = cs.get_default_configuration()
    X_train, y_train = train_data.data
    X_test, y_test = test_data.data
    print('X_train/test shapes: %s, %s' % (str(X_train.shape), str(X_test.shape)))

    # Build the ML estimator.
    from automlToolkit.components.evaluators.cls_evaluator import fetch_predict_estimator
    estimator = fetch_predict_estimator(default_config, X_train, y_train)

    y_pred = estimator.predict(X_test)
    print(balanced_accuracy(y_test, y_pred))
    print(balanced_accuracy(y_pred, y_test))


if __name__ == "__main__":
    classifier_id = 'liblinear_svc'
    dataset = 'pc4'
    evaluate_base_model(classifier_id, dataset)
