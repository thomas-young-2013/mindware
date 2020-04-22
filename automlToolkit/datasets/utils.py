import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from automlToolkit.utils.data_manager import DataManager
from automlToolkit.components.feature_engineering.fe_pipeline import FEPipeline
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.meta_learning.meta_features import calculate_all_metafeatures
from automlToolkit.components.utils.constants import MULTICLASS_CLS


def load_data(dataset, data_dir='./', datanode_returned=False, preprocess=True):
    dm = DataManager()
    data_path = data_dir + 'data/datasets/%s.csv' % dataset

    if dataset in ['credit_default']:
        data_path = data_dir + 'data/datasets/%s.xls' % dataset

    # Load train data.
    if dataset in ['higgs', 'amazon_employee', 'spectf', 'usps', 'vehicle_sensIT', 'codrna']:
        label_column = 0
    elif dataset in ['rmftsa_sleepdata(1)']:
        label_column = 1
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

    train_data_node = dm.load_train_csv(data_path, label_col=label_column, header=header, sep=sep,
                                        na_values=["n/a", "na", "--", "-", "?"])

    if preprocess:
        pipeline = FEPipeline(fe_enabled=False, metric='acc')
        train_data = pipeline.fit_transform(train_data_node)
    else:
        train_data = train_data_node

    if datanode_returned:
        return train_data
    else:
        X, y = train_data.data
        feature_types = train_data.feature_types
        return X, y, feature_types


def load_train_test_data(dataset, data_dir='./', test_size=0.2, random_state=45):
    X, y, feature_type = load_data(dataset, data_dir, False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    train_node = DataNode(data=[X_train, y_train], feature_type=feature_type.copy())
    test_node = DataNode(data=[X_test, y_test], feature_type=feature_type.copy())
    return train_node, test_node


def calculate_metafeatures(dataset, data_dir='./'):
    X, y, feature_types = load_data(dataset, data_dir, datanode_returned=False, preprocess=False)
    categorical = []
    for i in feature_types:
        if i == 'categorical':
            categorical.append(True)
        else:
            categorical.append(False)
    X = np.array(X)
    y = np.array(y)
    categorical_idx = []
    for idx, i in enumerate(categorical):
        if i:
            categorical_idx.append(idx)
    lbe = ColumnTransformer([('lbe', OrdinalEncoder(), categorical_idx)], remainder="passthrough")
    X = lbe.fit_transform(X).astype('float64')
    categorical_ = [True] * len(categorical_idx)
    categorical_false = [False] * (len(categorical) - len(categorical_idx))
    categorical_.extend(categorical_false)
    mf = calculate_all_metafeatures(X=X, y=y,
                                    categorical=categorical_,
                                    dataset_name=dataset,
                                    task_type=MULTICLASS_CLS)
    return mf.load_values()
