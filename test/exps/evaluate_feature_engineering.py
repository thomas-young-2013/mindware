import os
import sys
import pickle
import argparse
import numpy as np
import autosklearn.classification
sys.path.append(os.getcwd())
from automlToolkit.datasets.utils import load_data
from automlToolkit.components.evaluator import Evaluator
from automlToolkit.components.feature_engineering.fe_pipeline import FEPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=1200)
parser.add_argument('--datasets', type=str, default='dataset_small')
parser.add_argument('--mth', type=str, choices=['ausk', 'ours'], default='ours')
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
args = parser.parse_args()

proj_dir = './'
dataset_large = 'credit,diabetes,pc4,sick,spectf,splice,waveform,' \
                'winequality_red,winequality_white,ionosphere,amazon_employee,' \
                'lymphography,messidor_features,spambase,ap_omentum_ovary,a9a'
dataset_small = 'messidor_features,lymphography,winequality_red,winequality_white,credit,' \
                'ionosphere,splice,diabetes,pc4,spectf,spambase,amazon_employee'

if args.datasets == 'all':
    datasets = dataset_small.split(',')
else:
    datasets = args.datasets.split(',')
time_limit = args.time_limit
mth = args.mth


def evaluate_ausk_fe(dataset, time_limit, seed=1):
    print('==> Start to Evaluate', dataset, 'Budget', time_limit)
    from automlToolkit.utils.default_random_forest import DefaultRandomForest
    # Add random forest classifier (with default hyperparameter) component to auto-sklearn.
    autosklearn.pipeline.components.classification.add_classifier(DefaultRandomForest)
    include_models = ['DefaultRandomForest']

    # Construct the ML model.
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_limit,
        include_preprocessors=None,
        n_jobs=1,
        include_estimators=include_models,
        ensemble_memory_limit=8192,
        ml_memory_limit=8192,
        ensemble_size=1,
        ensemble_nbest=1,
        initial_configurations_via_metalearning=0,
        per_run_time_limit=600,
        seed=seed,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5}
    )
    print(automl)

    data_node = load_data(dataset, datanode_returned=True)

    X, y = data_node.data
    automl.fit(X.copy(), y.copy())
    model_desc = automl.show_models()
    print(model_desc)

    all_test_results = automl.cv_results_['mean_test_score']
    print('Mean test score', all_test_results)
    best_result = np.max(automl.cv_results_['mean_test_score'])
    print('Validation Accuracy', best_result)

    save_path = proj_dir + 'data/ausk_fe_%s_%d.pkl' % (dataset, time_limit)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, best_result], f)

    return best_result


def evaluate_evaluation_based_fe(dataset, time_limit, seed=1):
    # Prepare the configuration for random forest.
    from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
    from autosklearn.pipeline.components.classification.random_forest import RandomForest
    cs = RandomForest.get_hyperparameter_search_space()
    clf_hp = UnParametrizedHyperparameter("estimator", 'random_forest')
    cs.add_hyperparameter(clf_hp)
    evaluator = Evaluator(cs.get_default_configuration(), name='fe', seed=seed)

    raw_data = load_data(dataset, datanode_returned=True)

    pipeline = FEPipeline(fe_enabled=True, optimizer_type='eval_base',
                          time_budget=time_limit, evaluator=evaluator,
                          seed=seed, model_id='random_forest',
                          time_limit_per_trans=300
                          )
    train_data = pipeline.fit_transform(raw_data)

    score = evaluator(None, data_node=train_data)
    print('==> Base validation score', score)

    save_path = proj_dir + 'data/fe_%s_%d.pkl' % (dataset, time_limit)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, score], f)
    return score


if __name__ == "__main__":
    for dataset in datasets:
        if mth == 'ausk':
            evaluate_ausk_fe(dataset, time_limit)
        else:
            evaluate_evaluation_based_fe(dataset, time_limit)
