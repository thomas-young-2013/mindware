from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.datasets.utils import load_data
from automlToolkit.components.evaluator import Evaluator
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier
from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC

raw_data = load_data('diabetes', datanode_returned=True)

cs = LibLinear_SVC.get_hyperparameter_search_space()
model = UnParametrizedHyperparameter("estimator", 'liblinear_svc')
cs.add_hyperparameter(model)

evaluator = Evaluator(cs.get_default_configuration(), name='hpo', data_node=raw_data)
optimizer = SMACOptimizer(evaluator, cs, trials_per_iter=10)

results = list()
for iter in range(20):
    perf, _, _ = optimizer.iterate()
    print(iter, perf)
    results.append(perf)
    print(iter, results)
