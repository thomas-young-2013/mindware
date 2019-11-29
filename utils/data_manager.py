import numpy as np
import pandas as pd

from components.utils.constants import *
from components.utils.utils import is_discrete, detect_abnormal_type

default_missing_values = ["n/a", "na", "--", "-", "?"]


class DataManager(object):
    """
    This class implements the wrapper for data used in the ML task.

    It finishes the following preprocesses:
    1) detect the type of each feature (numerical, categorical, textual, ...)
    """

    # X,y should be None if using DataManager().load_csv(...)
    def __init__(self, X=None, y=None, na_values=default_missing_values):
        self.na_values = na_values
        self.feature_types = None
        self.missing_flags = None
        self.train_X, self.train_y = None, None
        self.test_X, self.test_y = None, None
        self.label_name = None

        if X is not None:
            self.train_X = np.array(X)
            self.train_y = np.array(y)
            self.set_feat_types(pd.DataFrame(self.train_X), [])

    def set_feat_types(self, df, columns_missed):
        self.missing_flags = list()
        for idx, col_name in enumerate(df.columns):
            self.missing_flags.append(True if col_name in columns_missed else False)

        self.feature_types = list()
        for idx, col_name in enumerate(df.columns):
            col_vals = df[col_name].values
            dtype = df[col_name].dtype

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

    def clean_data_with_nan(self, df, label_col, phase='train', drop_index=None, has_label=True):
        columns_missed = df.columns[df.isnull().any()].tolist()

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

        if drop_index:
            drop_col = [df.columns[index] for index in drop_index]
            df.drop(drop_col, axis=1, inplace=True)

        # Delete the label column.
        if has_label:
            df.drop(label_colname, axis=1, inplace=True)

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
        columns_missed = df.columns[df.isnull().any()].tolist()

        self.clean_data_with_nan(df, label_col, drop_index=drop_index)

        # set the feature types
        self.set_feat_types(df, columns_missed)
        self.train_X = df
        return df, self.train_y

    def load_test_csv(self, file_location, has_label=False, label_col=-1,
                      drop_index=None, keep_default_na=True, header='infer',
                      sep=','):
        df = pd.read_csv(file_location, keep_default_na=keep_default_na,
                         na_values=self.na_values, header=header, sep=sep)
        # Drop the row with all NaNs.
        df.dropna(how='all')
        columns_missed = df.columns[df.isnull().any()].tolist()
        self.clean_data_with_nan(df, label_col, phase='test', drop_index=drop_index, has_label=has_label)
        self.set_feat_types(df, columns_missed)
        self.test_X = df
        return df
