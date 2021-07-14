import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing._encoders import OrdinalEncoder, _BaseEncoder
from mindware.utils.data_manager import DataManager
from mindware.components.feature_engineering.fe_pipeline import FEPipeline
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.meta_learning.meta_feature.meta_features import calculate_all_metafeatures
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS


def load_data(dataset, data_dir='./', datanode_returned=False, preprocess=True, task_type=None):
    dm = DataManager()
    if task_type is None:
        data_path = data_dir + 'data/datasets/%s.csv' % dataset
    elif task_type in CLS_TASKS:
        data_path = data_dir + 'data/cls_datasets/%s.csv' % dataset
    elif task_type in RGS_TASKS:
        data_path = data_dir + 'data/rgs_datasets/%s.csv' % dataset
    else:
        raise ValueError("Unknown task type %s" % str(task_type))

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
        pipeline = FEPipeline(fe_enabled=False, metric='acc', task_type=task_type)
        train_data = pipeline.fit_transform(train_data_node)
    else:
        train_data = train_data_node

    if datanode_returned:
        return train_data
    else:
        X, y = train_data.data
        feature_types = train_data.feature_types
        return X, y, feature_types


def load_train_test_data(dataset, data_dir='./', test_size=0.2, task_type=None, random_state=45):
    X, y, feature_type = load_data(dataset, data_dir, False, task_type=task_type)
    if task_type is None or task_type in CLS_TASKS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    train_node = DataNode(data=[X_train, y_train], feature_type=feature_type.copy())
    test_node = DataNode(data=[X_test, y_test], feature_type=feature_type.copy())
    # print('is imbalanced dataset', is_imbalanced_dataset(train_node))
    return train_node, test_node


nan_replace = 'unknown_nan'


class NanOrdinalEncoder(_BaseEncoder):
    def __init__(self):
        self.lbe = OrdinalEncoder()

    def fit(self, X):
        self.lbe.fit(X)
        return self

    def transform(self, X):
        result = self.lbe.transform(X)
        for col in range(result.shape[1]):
            nan_idx = list(self.lbe.categories_[col]).index(nan_replace)
            column = result[:, col]
            column[column == nan_idx] = np.nan
        return result


def calculate_metafeatures(dataset, dataset_id=None, data_dir='./', task_type=None):
    if isinstance(dataset, str):
        X, y, feature_types = load_data(dataset, data_dir, datanode_returned=False, preprocess=False,
                                        task_type=task_type)
        dataset_id = dataset
    elif isinstance(dataset, DataNode):
        X, y, feature_types = dataset.data[0], dataset.data[1], dataset.feature_types
        import pandas as pd
        X = pd.DataFrame(data=X)
    else:
        raise ValueError('Invalid dataset input!')

    categorical_idx = [i for i, feature_type in enumerate(feature_types) if feature_type == 'categorical']

    nan_val = np.array(X.isnull()).astype('int')
    nan_avg = np.average(nan_val, axis=0)
    nan_idx = [idx for idx in range(len(nan_avg)) if nan_avg[idx] != 0]
    nan_categorical_idx = list(set(nan_idx).intersection(categorical_idx))

    for col in X.columns[nan_categorical_idx]:
        X[col].fillna(nan_replace, inplace=True)
    X = np.array(X)
    y = np.array(y)
    normal_categorical_idx = list(set(categorical_idx) - set(nan_categorical_idx))
    lbe = ColumnTransformer([('lbe', OrdinalEncoder(), normal_categorical_idx),
                             ('nan_lbe', NanOrdinalEncoder(), nan_categorical_idx)],
                            remainder="passthrough")
    X = lbe.fit_transform(X).astype('float64')
    categorical_ = [True] * len(categorical_idx)
    categorical_false = [False] * (len(feature_types) - len(categorical_idx))
    categorical_.extend(categorical_false)
    mf = calculate_all_metafeatures(X=X, y=y,
                                    categorical=categorical_,
                                    dataset_name=dataset_id,
                                    task_type=task_type)
    return mf.load_values()
