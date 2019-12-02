from automlToolkit.utils.data_manager import DataManager
from automlToolkit.components.feature_engineering.fe_pipeline import FEPipeline


def load_data(dataset, proj_dir='./', datanode_returned=False):
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
    if datanode_returned:
        return train_data
    else:
        X, y = train_data.data
        feature_types = train_data.feature_types
        return X, y, feature_types
