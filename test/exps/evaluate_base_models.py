import os
import sys
import time
import pickle
import argparse
import numpy as np
import autosklearn.classification
from tabulate import tabulate

sys.path.append(os.getcwd())

from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.cls_metrics import balanced_accuracy
from solnml.components.evaluators.evaluator import Evaluator
from solnml.utils.logging_utils import get_logger
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from solnml.components.optimizers.smac_optimizer import SMACOptimizer
from solnml.components.optimizers.psmac_optimizer import PSMACOptimizer
from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer
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
    from solnml.components.evaluators.cls_evaluator import fetch_predict_estimator
    estimator = fetch_predict_estimator(default_config, X_train, y_train)

    y_pred = estimator.predict(X_test)
    print(balanced_accuracy(y_test, y_pred))
    print(balanced_accuracy(y_pred, y_test))


def evaluate(train_data, test_data, config):
    X_train, y_train = train_data.data
    X_test, y_test = test_data.data
    print('X_train/test shapes: %s, %s' % (str(X_train.shape), str(X_test.shape)))

    # Build the ML estimator.
    from solnml.components.evaluators.evaluator import fetch_predict_estimator
    estimator = fetch_predict_estimator(config, X_train, y_train)

    y_pred = estimator.predict(X_test)
    return balanced_accuracy(y_test, y_pred)


def evaluate_issue_source(classifier_id, dataset, opt_type='hpo'):
    _start_time = time.time()
    train_data, test_data = load_train_test_data(dataset)

    from autosklearn.pipeline.components.classification import _classifiers
    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)
    default_config = cs.get_default_configuration()

    seed = 2343
    if opt_type == 'hpo':
        evaluator = Evaluator(default_config,
                              data_node=train_data, name='hpo',
                              resampling_strategy='holdout',
                              seed=seed)
        optimizer = SMACOptimizer(
                evaluator, cs, output_dir='logs/', per_run_time_limit=300,
                trials_per_iter=5, seed=seed)
    else:
        evaluator = Evaluator(default_config,
                              name='fe', resampling_strategy='holdout',
                              seed=seed)
        optimizer = EvaluationBasedOptimizer(
            'classification',
            train_data, evaluator,
            classifier_id, 300, 1024, seed)

    perf_result = list()
    for iter_id in range(20):
        optimizer.iterate()
        print('='*30)
        print('ITERATION: %d' % iter_id)
        if opt_type == 'hpo':
            config = optimizer.incumbent_config
            perf = evaluate(train_data, test_data, config)
        else:
            fe_train_data = optimizer.incumbent
            fe_test_data = optimizer.apply(test_data, fe_train_data)
            perf = evaluate(fe_train_data, fe_test_data, default_config)
        print(perf)
        print('='*30)
        perf_result.append(perf)

    print(perf_result)


if __name__ == "__main__":
    classifier_id = 'liblinear_svc'
    dataset = 'pc4'
    # evaluate_base_model(classifier_id, dataset)
    evaluate_issue_source(classifier_id, dataset, opt_type='fe')
