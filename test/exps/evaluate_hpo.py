import os
import sys
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
sys.path.append(os.getcwd())
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.datasets.utils import load_data
from automlToolkit.components.evaluator import Evaluator


def conduct_hpo(dataset='pc4', classifier_id='random_forest', iter_mode=True):
    from autosklearn.pipeline.components.classification import _classifiers

    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)

    raw_data = load_data(dataset, datanode_returned=True)
    evaluator = Evaluator(cs.get_default_configuration(), name='hpo', data_node=raw_data)

    if not iter_mode:
        optimizer = SMACOptimizer(evaluator, cs, evaluation_limit=300, output_dir='logs')
        inc, val = optimizer.optimize()
        print(inc, val)
    else:
        import time
        _start_time = time.time()
        optimizer = SMACOptimizer(
            evaluator, cs, trials_per_iter=1, output_dir='logs')
        results = list()
        for _iter in range(50):
            perf, _, _ = optimizer.iterate()
            print(_iter, perf)
            results.append(perf)
        print(results)
        print(time.time() - _start_time)


if __name__ == "__main__":
    # test_case: amazon_employee, gradient_boosting.
    # 2: 113.09
    # 20: 642.82
    # None: 2956.58
    conduct_hpo(dataset='winequality_white', classifier_id='gradient_boosting')
