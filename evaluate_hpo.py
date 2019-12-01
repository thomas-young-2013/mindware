from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.components.evaluator import Evaluator
from evaluate_transgraph import engineer_data

raw_data, _ = engineer_data('pc4', 'none')

from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier

cs = AdaboostClassifier.get_hyperparameter_search_space()
model = UnParametrizedHyperparameter("estimator", 'adaboost')
cs.add_hyperparameter(model)

evaluator = Evaluator(cs.get_default_configuration(), data_node=raw_data)
evaluator(None)
optimizer = SMACOptimizer(evaluator, cs)

for iter in range(20):
    perf, _ = optimizer.iterate()
    print(iter, perf)
