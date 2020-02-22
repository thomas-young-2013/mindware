import os
import sys
import argparse
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer

sys.path.append(os.getcwd())
from automlToolkit.utils.data_manager import DataManager
from automlToolkit.components.evaluators.reg_evaluator import RegressionEvaluator
from automlToolkit.components.feature_engineering.fe_pipeline import FEPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=1200)
parser.add_argument('--data_dir', type=str, default='/Users/thomasyoung/PycharmProjects/AI_anti_plague/')
args = parser.parse_args()
proj_dir = './'
time_limit = args.time_limit
data_dir = args.data_dir


def smape(y_true, y_pred):
    sample_size = len(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2 + 1e-6
    return np.sum(np.divide(numerator, denominator)) / sample_size


class LightGBMRegressor():
    def __init__(self, n_estimators, learning_rate, num_leaves, min_child_weight,
                 subsample, colsample_bytree, reg_alpha, reg_lambda):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree

        self.n_jobs = -1
        self.estimator = None

    def fit(self, X, y, metric=smape):
        self.estimator = LGBMRegressor(num_leaves=self.num_leaves,
                                       learning_rate=self.learning_rate,
                                       n_estimators=self.n_estimators,
                                       objective='regression',
                                       min_child_weight=self.min_child_weight,
                                       subsample=self.subsample,
                                       colsample_bytree=self.colsample_bytree,
                                       reg_alpha=self.reg_alpha,
                                       reg_lambda=self.reg_lambda,
                                       n_jobs=self.n_jobs,
                                       silent=False
                                       )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


def create_csv(task_id=3):
    import pandas as pd
    train_data = pd.read_csv(data_dir + 'data/candidate_train.csv')
    train_answer = pd.read_csv(data_dir + 'data/train_answer.csv')
    test_data = pd.read_csv(data_dir + 'data/candidate_val.csv')

    print('complete dataset loading...')
    train_data = train_data.merge(train_answer, on='id', how='left')
    train_raw_data = train_data.iloc[:, 0:3177]
    test_raw_data = test_data.iloc[:, 0:3177]
    label = train_data['p%d' % task_id]
    result = pd.concat([train_raw_data, label], axis=1).reset_index(drop=True)
    result.to_csv(data_dir + 'data/p%s.csv' % task_id, index=False)
    print(result.head())
    test_raw_data.to_csv(data_dir + 'data/test_data.csv', index=False)
    print(test_raw_data.head())


def preprocess_data(task_id=3):
    dm = DataManager()
    data_path = data_dir + 'data/p%d.csv' % task_id
    if not os.path.exists(data_path):
        create_csv(task_id)
    print('load train csv.')
    dm.load_train_csv(data_path, label_col=-1, header='infer', sep=',')
    pipeline = FEPipeline(fe_enabled=False)
    raw_data = pipeline.fit_transform(dm)
    print('raw data processing finished.')
    return raw_data


def test_evaluator():
    config = {'colsample_bytree': 0.7214005546233202,
              'estimator': 'lightgbm',
              'learning_rate': 0.20740875048979773,
              'min_child_weight': 5,
              'n_estimators': 424,
              'num_leaves': 82,
              'reg_alpha': 0.001268145413023973,
              'reg_lambda': 0.15002466116267585,
              'subsample': 0.8110820196868197}
    config.pop('estimator', None)
    gbm = LightGBMRegressor(**config)
    scorer = make_scorer(smape, greater_is_better=False)
    raw_data = preprocess_data()
    evaluator = RegressionEvaluator(None, scorer, data_node=raw_data, name='fe', seed=1, estimator=gbm)
    print(evaluator(None))


def evaluation_based_feature_engineering(time_limit, seed=1):
    config = {'colsample_bytree': 0.7214005546233202,
              'estimator': 'lightgbm',
              'learning_rate': 0.20740875048979773,
              'min_child_weight': 5,
              'n_estimators': 424,
              'num_leaves': 82,
              'reg_alpha': 0.001268145413023973,
              'reg_lambda': 0.15002466116267585,
              'subsample': 0.8110820196868197}
    config.pop('estimator', None)
    gbm = LightGBMRegressor(**config)
    scorer = make_scorer(smape, greater_is_better=False)
    evaluator = RegressionEvaluator(None, scorer, name='fe', seed=seed, estimator=gbm)
    raw_data = preprocess_data()
    pipeline = FEPipeline(task_type='regression', task_id='anti_plague',
                          fe_enabled=True, optimizer_type='eval_base',
                          time_budget=time_limit, evaluator=evaluator,
                          seed=seed, model_id='lightgbm',
                          time_limit_per_trans=300
                          )
    train_data = pipeline.fit_transform(raw_data)
    print(train_data.score)


if __name__ == "__main__":
    test_evaluator()
    # evaluation_based_feature_engineering(time_limit)
