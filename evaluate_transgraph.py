import os
import sys
import argparse
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.feature_engineering.fe_pipeline import FEPipeline

proj_dir = '/home/thomas/PycharmProjects/Feature-Engineering/'
if not os.path.exists(proj_dir):
    proj_dir = './'

sys.path.append(proj_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='credit')
args = parser.parse_args()


def load_data(dataset):
    from automlToolkit.utils.data_manager import DataManager
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


def engineer_data(dataset, fe='none', evaluator=None, time_budget=None, seed=1):
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
        pipeline = FEPipeline(fe_enabled=True, optimizer_type='eval_base',
                              time_budget=time_budget, evaluator=evaluator, seed=seed)
        train_data = pipeline.fit_transform(input_data)
    else:
        raise ValueError('Invalid method: %s' % fe)

    print('Finish feature processing.')
    print('train data', train_data)
    return train_data, time.time() - start_time
