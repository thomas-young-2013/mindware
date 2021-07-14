import numpy as np
import pandas as pd

from mindware.components.utils.constants import *
from mindware.components.utils.utils import is_discrete, detect_abnormal_type, detect_categorical_type
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.feature_engineering.transformations.preprocessor.imputer import ImputationTransformation
from mindware.components.feature_engineering.transformations.preprocessor.onehot_encoder import \
    OneHotTransformation
from mindware.components.feature_engineering.transformations.selector.variance_selector import VarianceSelector

default_missing_values = ["n/a", "na", "--", "-", "?"]


class DataManager(object):
    """
    This class implements the wrapper for data used in the ML task.

    It finishes the following preprocesses:
    1) detect the type of each feature (numerical, categorical, textual, ...)
    """

    # X,y should be None if using DataManager().load_csv(...)
    def __init__(self, X=None, y=None, na_values=default_missing_values, feature_types=None, feature_names=None):
        self.na_values = na_values
        self.feature_types = feature_types
        self.feature_names = feature_names
        self.missing_flags = None
        self.train_X, self.train_y = None, None
        self.test_X, self.test_y = None, None
        self.label_name = None

        if X is not None:
            self.train_X = np.array(X)
            self.train_y = np.array(y)
            if feature_types is None:
                self.set_feat_types(pd.DataFrame(self.train_X), [])

    def set_feat_types(self, df, columns_missed):
        self.missing_flags = list()
        for idx, col_name in enumerate(df.columns):
            self.missing_flags.append(True if col_name in columns_missed else False)

        self.feature_types = list()
        for idx, col_name in enumerate(df.columns):
            col_vals = df[col_name].values
            dtype = df[col_name].dtype

            # Filter the element with missing value.
            cleaned_vals = col_vals
            if col_name in columns_missed:
                cleaned_vals = np.array([val for val in col_vals if not pd.isnull(val)])

            if dtype in [np.int, np.int16, np.int32, np.int64]:
                feat_type = DISCRETE
            elif dtype in [np.float, np.float16, np.float32, np.float64, np.double]:
                feat_type = DISCRETE if is_discrete(cleaned_vals) else NUMERICAL
            else:
                flag, cand_values, ab_idx, is_str = detect_abnormal_type(col_vals)
                if flag:
                    # Set the invalid element to NaN.
                    df.at[ab_idx, col_name] = np.nan
                    # Refresh the cleaned column.
                    cleaned_vals = np.array([val for val in df[col_name].values if not pd.isnull(val)])
                    if is_str:
                        feat_type = CATEGORICAL
                    else:
                        feat_type = DISCRETE if is_discrete(cleaned_vals) else NUMERICAL
                else:
                    feat_type = CATEGORICAL
            self.feature_types.append(feat_type)

    def get_data_node(self, X, y):
        if self.feature_types is None:
            raise ValueError("Feature type missing")
        return DataNode([X, y], self.feature_types, feature_names=self.feature_names)

    def clean_data_with_nan(self, df, label_col, phase='train', drop_index=None, has_label=True):
        columns_missed = df.columns[df.isnull().any()].tolist()

        if has_label:
            if self.label_name is None:
                if phase != 'train':
                    print('Warning: Label is not specified! set label_col=%d by default.' % label_col)
                label_colname = df.columns[label_col]
            else:
                label_colname = self.label_name

            self.label_name = label_colname
            if label_colname in columns_missed:
                labels = df[label_colname].values
                row_idx = [idx for idx, val in enumerate(labels) if np.isnan(val)]
                # Delete the row with NaN label.
                df.drop(df.index[row_idx], inplace=True)

            if phase == 'train':
                self.train_y = df[label_colname].values
            else:
                self.test_y = df[label_colname].values

            # Delete the label column.
            df.drop(label_colname, axis=1, inplace=True)

        if drop_index:
            drop_col = [df.columns[index] for index in drop_index]
            df.drop(drop_col, axis=1, inplace=True)

    def load_train_csv(self, file_location, label_col=-1, drop_index=None,
                       keep_default_na=True, na_values=None, header='infer',
                       sep=','):
        # Set the NA values.
        if na_values is not None:
            na_set = set(self.na_values)
            for item in na_values:
                na_set.add(item)
            self.na_values = list(na_set)

        if file_location.endswith('csv'):
            df = pd.read_csv(file_location, keep_default_na=keep_default_na,
                             na_values=self.na_values, header=header, sep=sep)
        elif file_location.endswith('xls'):
            df = pd.read_csv(file_location, keep_default_na=keep_default_na,
                             na_values=self.na_values, header=header)
        else:
            raise ValueError('Unsupported file format: %s!' % file_location.split('.')[-1])

        # Drop the row with all NaNs.
        df.dropna(how='all')

        # Clean the data where the label columns have nans.
        self.clean_data_with_nan(df, label_col, drop_index=drop_index)

        # The columns with missing values.
        columns_missed = df.columns[df.isnull().any()].tolist()

        # Identify the feature types
        self.set_feat_types(df, columns_missed)

        self.train_X = df
        data = [self.train_X, self.train_y]
        return DataNode(data, self.feature_types, feature_names=self.train_X.columns.values)

    def load_test_csv(self, file_location, has_label=False, label_col=-1,
                      drop_index=None, keep_default_na=True, header='infer',
                      sep=','):
        df = pd.read_csv(file_location, keep_default_na=keep_default_na,
                         na_values=self.na_values, header=header, sep=sep)
        # Drop the row with all NaNs.
        df.dropna(how='all')
        self.clean_data_with_nan(df, label_col, phase='test', drop_index=drop_index, has_label=has_label)
        self.test_X = df

        data = [self.test_X, self.test_y]
        return DataNode(data, self.feature_types, feature_names=self.test_X.columns.values)

    def preprocess(self, input_node, task_type=CLASSIFICATION, train_phase=True):
        try:
            input_node = self.remove_uninf_cols(input_node, train_phase=True)
            input_node = self.impute_cols(input_node)
            input_node = self.one_hot(input_node)
        except AttributeError as e:
            print('data[0] in input_node should be a DataFrame!')
        input_node = self.remove_cols_with_same_values(input_node)
        # print('=' * 20)
        if self.task_type is None or self.task_type in CLS_TASKS:
            # Label encoding.
            input_node = self.encode_label(input_node)
        return input_node

    def preprocess_fit(self, input_node, task_type=CLASSIFICATION):
        self.task_type = task_type
        self.variance_selector = None
        self.onehot_encoder = None
        self.label_encoder = None
        preprocessed_node = self.preprocess(input_node, train_phase=True)
        return preprocessed_node

    def preprocess_transform(self, input_node):
        preprocessed_node = self.preprocess(input_node, train_phase=False)
        return preprocessed_node

    def remove_uninf_cols(self, input_node: DataNode, train_phase=True):
        raw_dataframe = input_node.data[0]
        types = input_node.feature_types
        if train_phase:
            # Remove the uninformative columns.
            uninformative_columns, uninformative_idx = list(), list()
            for idx, column in enumerate(list(raw_dataframe)):
                if raw_dataframe[column].isnull().values.all():
                    uninformative_columns.append(column)
                    uninformative_idx.append(idx)
                    continue
                if types[idx] == CATEGORICAL:
                    num_sample = input_node.data[0].shape[0]
                    num_unique = len(set(input_node.data[0][column]))
                    if num_unique >= int(0.8 * num_sample):
                        uninformative_columns.append(column)
                        uninformative_idx.append(idx)
            self.uninformative_columns, self.uninformative_idx = uninformative_columns, uninformative_idx

        input_node.feature_types = [types[idx] for idx in range(len(types)) if idx not in self.uninformative_idx]
        raw_dataframe = raw_dataframe.drop(self.uninformative_columns, axis=1)
        input_node.data[0] = raw_dataframe
        return input_node

    def impute_cols(self, input_node: DataNode):
        raw_dataframe = input_node.data[0]
        feat_types = input_node.feature_types
        need_imputation = raw_dataframe.isnull().values.any()
        if need_imputation:
            for idx, column in enumerate(list(raw_dataframe)):
                if raw_dataframe[column].isnull().values.any():
                    feature_type = feat_types[idx]
                    if feature_type in [CATEGORICAL, ORDINAL]:
                        imputer = ImputationTransformation('most_frequent')
                        input_node = imputer.operate(input_node, [idx])
                    else:
                        imputer = ImputationTransformation('median')
                        input_node = imputer.operate(input_node, [idx])
        return input_node

    def one_hot(self, input_node: DataNode):
        # One-hot encoding TO categorical features.
        categorical_fields = [idx for idx, type in enumerate(input_node.feature_types) if type == CATEGORICAL]
        if len(categorical_fields) > 0:
            if self.onehot_encoder is None:
                self.onehot_encoder = OneHotTransformation()
            input_node = self.onehot_encoder.operate(input_node, categorical_fields)
        return input_node

    def remove_cols_with_same_values(self, input_node: DataNode):
        if self.variance_selector is None:
            self.variance_selector = VarianceSelector()
        input_node = self.variance_selector.operate(input_node)
        return input_node

    def encode_label(self, input_node: DataNode):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        X, y = input_node.data
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(y)
            y = self.label_encoder.transform(y)
        input_node.data = (X, y)
        return input_node
