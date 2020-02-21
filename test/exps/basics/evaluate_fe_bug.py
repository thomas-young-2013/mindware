import os
import sys
import argparse
import numpy as np

sys.path.append(os.getcwd())

from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from autosklearn.pipeline.components.classification import _classifiers

from automlToolkit.datasets.utils import load_train_test_data
from automlToolkit.components.evaluators.evaluator import Evaluator
from automlToolkit.components.feature_engineering.fe_pipeline import FEPipeline

parser = argparse.ArgumentParser()
dataset_set = 'yeast,diabetes,vehicle,spectf,credit,' \
              'ionosphere,lymphography,messidor_features,winequality_red'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--rep_num', type=int, default=10)
parser.add_argument('--time_limit', type=int, default=600)


per_run_time_limit = 150


def evaluate_fe_bugs(dataset, run_id, time_limit, seed):
    algorithms = ['lda', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                  'adaboost', 'random_forest', 'extra_trees', 'decision_tree']
    algo_id = np.random.choice(algorithms, 1)[0]
    task_id = '%s-fe-%s-%d' % (dataset, algo_id, run_id)
    print(task_id)

    # Prepare the configuration for random forest.
    clf_class = _classifiers[algo_id]
    cs = clf_class.get_hyperparameter_search_space()
    clf_hp = UnParametrizedHyperparameter("estimator", algo_id)
    cs.add_hyperparameter(clf_hp)
    evaluator = Evaluator(cs.get_default_configuration(),
                          name='fe', seed=seed,
                          resampling_strategy='holdout')

    pipeline = FEPipeline(fe_enabled=True, optimizer_type='eval_base',
                          time_budget=time_limit, evaluator=evaluator,
                          seed=seed, model_id=algo_id,
                          time_limit_per_trans=per_run_time_limit,
                          task_id=task_id
                          )

    raw_data, test_raw_data = load_train_test_data(dataset)
    train_data = pipeline.fit_transform(raw_data.copy_())
    test_data = pipeline.transform(test_raw_data.copy_())
    train_data_new = pipeline.transform(raw_data.copy_())

    assert (train_data.data[0] == train_data_new.data[0]).all()
    assert (train_data.data[1] == train_data_new.data[1]).all()
    assert (train_data_new == train_data)

    score = evaluator(None, data_node=test_data)
    print('==> Test score', score)


if __name__ == "__main__":
    args = parser.parse_args()
    rep = args.rep_num
    time_limit = args.time_limit
    for dataset in args.datasets.split(','):
        for run_id in range(rep):
            evaluate_fe_bugs(dataset, run_id, time_limit, seed=run_id)
