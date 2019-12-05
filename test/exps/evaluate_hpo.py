import os
import sys
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier
from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC
from autosklearn.pipeline.components.classification.libsvm_svc import LibSVM_SVC
sys.path.append(os.getcwd())
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.datasets.utils import load_data
from automlToolkit.components.evaluator import Evaluator

raw_data = load_data('pc4', datanode_returned=True)

cs = LibSVM_SVC.get_hyperparameter_search_space()
model = UnParametrizedHyperparameter("estimator", 'libsvm_svc')
cs.add_hyperparameter(model)

evaluator = Evaluator(cs.get_default_configuration(), name='hpo', data_node=raw_data)


mode = 2
# libsvm_svc: 0.707 vs 0.701(iter,ori), credit.
# libsvm_svc: 0.8786 vs .8786(iter,ori), pc4.

if mode == 1:
    optimizer = SMACOptimizer(evaluator, cs, evaluation_limit=150)
    inc, val = optimizer.optimize()
    print(inc, val)
else:
    optimizer = SMACOptimizer(evaluator, cs, trials_per_iter=5)
    results = list()
    for iter in range(30):
        perf, _, _ = optimizer.iterate()
        print(iter, perf)
        results.append(perf)

    print(results)
