import os
import sys
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from fe_components.transformation_graph import DataNode
from fe_components.fe_pipeline import FEPipeline

proj_dir = '/home/thomas/PycharmProjects/Feature-Engineering/'
if not os.path.exists(proj_dir):
    proj_dir = './'

sys.path.append(proj_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='credit')
args = parser.parse_args()


def evaluate_fe_pipeline():
    from utils.data_manager import DataManager
    dm = DataManager()
    data_path = 'data/datasets/pc4.csv'

    # Load train data.
    dm.load_train_csv(data_path)
    print(dm.train_X.shape, dm.train_y.shape)

    pipeline = FEPipeline()
    train_data = pipeline.fit_transform(dm)
    print(train_data.data[0].shape)
    print(set(train_data.data[1]))

    # Load test data.
    dm.load_test_csv(data_path, has_label=True)
    print(dm.test_X.shape, dm.test_y.shape)
    test_data = pipeline.transform(dm)

    assert (train_data.data[0] == test_data.data[0]).all()


def load_data(dataset):
    from utils.data_manager import DataManager
    dm = DataManager()
    data_path = proj_dir + 'data/datasets/%s.csv' % dataset

    if dataset in ['credit_default']:
        data_path = proj_dir + 'data/datasets/%s.xls' % dataset

    # Load train data.
    if dataset in ['higgs', 'amazon_employee', 'spectf']:
        label_column = 0
    else:
        label_column = -1

    if dataset in ['spambase', 'messidor_features']:
        header = None
    else:
        header = 'infer'

    if dataset in ['winequality_white', 'winequality_red']:
        sep = ';'
    else:
        sep = ','

    dm.load_train_csv(data_path, label_col=label_column, header=header, sep=sep)

    pipeline = FEPipeline(fe_enabled=False)
    train_data = pipeline.fit_transform(dm)
    X, y = train_data.data
    feature_types = train_data.feature_types
    return X, y, feature_types


def engineer_data(dataset, fe='none', time_budget=None, seed=1):
    import time
    start_time = time.time()
    X, y, feature_types = load_data(dataset)
    input_data = DataNode(data=[X, y], feature_type=feature_types)

    if fe == 'none':
        train_data = input_data
    elif fe == 'epd_rdc':
        pipeline = FEPipeline(fe_enabled=True, optimizer_type='epd_rdc', seed=seed)
        train_data = pipeline.fit_transform(input_data)
    elif fe == 'eval_base':
        from evaluator import Evaluator
        from utils.default_random_forest import DefaultRandomForest
        cs = DefaultRandomForest.get_hyperparameter_search_space()
        config = cs.get_default_configuration().get_dictionary()
        clf = DefaultRandomForest(**config, random_state=seed)
        evaluator = Evaluator(seed=seed, clf=clf)
        pipeline = FEPipeline(fe_enabled=True, optimizer_type='eval_base',
                              time_budget=time_budget, evaluator=evaluator, seed=seed)
        train_data = pipeline.fit_transform(input_data)
    else:
        raise ValueError('Invalid method: %s' % fe)

    print('Finish feature processing.')
    print('train data', train_data)
    return train_data, time.time() - start_time


def evaluate_ml_baseline():
    def eval(dataset, fe='none', seed=1):
        np.random.seed(seed)

        from sklearn.model_selection import cross_val_score
        train_data, _ = engineer_data(dataset, fe, time_budget=300)

        X_, y_ = train_data.data
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        score = cross_val_score(clf, X_, y_, cv=5).mean()
        print('==> Validation score', score)

    for dataset in args.datasets.split(','):
        # eval(dataset)
        # eval(dataset, fe='epd_rdc')
        eval(dataset, fe='eval_base')


if __name__ == "__main__":
    # evaluate_transformations()
    # evaluate_fe_pipeline()
    # evaluate_data_manager()
    evaluate_ml_baseline()
