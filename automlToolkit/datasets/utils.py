from sklearn.model_selection import train_test_split
from automlToolkit.utils.data_manager import DataManager
from automlToolkit.components.feature_engineering.fe_pipeline import FEPipeline
from automlToolkit.components.feature_engineering.transformation_graph import DataNode


def load_data(dataset, proj_dir='./', datanode_returned=False):
    dm = DataManager()
    data_path = proj_dir + 'data/datasets/%s.csv' % dataset

    if dataset in ['credit_default']:
        data_path = proj_dir + 'data/datasets/%s.xls' % dataset

    # Load train data.
    if dataset in ['higgs', 'amazon_employee', 'spectf', 'usps']:
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
    if datanode_returned:
        return train_data
    else:
        X, y = train_data.data
        feature_types = train_data.feature_types
        return X, y, feature_types


def load_train_test_data(dataset, proj_dir='./', test_size=0.2, random_state=45):
    X, y, feature_type = load_data(dataset, proj_dir, False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    train_node = DataNode(data=[X_train, y_train], feature_type=feature_type.copy())
    test_node = DataNode(data=[X_test, y_test], feature_type=feature_type.copy())
    return train_node, test_node
