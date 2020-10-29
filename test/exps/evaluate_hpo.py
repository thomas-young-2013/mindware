import os
import sys
import argparse
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
sys.path.append(os.getcwd())
from solnml.components.optimizers.smac_optimizer import SMACOptimizer
from solnml.datasets.utils import load_data
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--dataset', type=str, default='spectf')
parser.add_argument('--algo', type=str, default='gradient_boosting')
parser.add_argument('--iter_num', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)


def conduct_hpo(dataset='pc4', classifier_id='random_forest', iter_num=100, iter_mode=True):
    from autosklearn.pipeline.components.classification import _classifiers

    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)

    raw_data = load_data(dataset, datanode_returned=True)
    print(set(raw_data.data[1]))
    evaluator = ClassificationEvaluator(cs.get_default_configuration(), name='hpo', data_node=raw_data)

    if not iter_mode:
        optimizer = SMACOptimizer(evaluator, cs, evaluation_limit=600, output_dir='logs')
        inc, val = optimizer.optimize()
        print(inc, val)
    else:
        import time
        _start_time = time.time()
        optimizer = SMACOptimizer(
            evaluator, cs, trials_per_iter=1,
            output_dir='logs', per_run_time_limit=180
        )
        results = list()
        for _iter in range(iter_num):
            perf, _, _ = optimizer.iterate()
            print(_iter, perf)
            results.append(perf)
        print(results)
        print(time.time() - _start_time)


if __name__ == "__main__":
    args = parser.parse_args()
    conduct_hpo(dataset=args.dataset, classifier_id=args.algo, iter_num=args.iter_num)
