import numpy as np
import warnings
import os
import pickle as pkl
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics.scorer import _BaseScorer

from automlToolkit.components.ensemble.base_ensemble import BaseEnsembleModel
from automlToolkit.components.utils.constants import CLS_TASKS
from automlToolkit.components.evaluators.base_evaluator import fetch_predict_estimator


class Stacking(BaseEnsembleModel):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None,
                 meta_learner='xgboost',
                 kfold=5):
        super().__init__(stats=stats,
                         ensemble_method='blending',
                         ensemble_size=ensemble_size,
                         task_type=task_type,
                         metric=metric,
                         output_dir=output_dir)

        self.kfold = kfold
        try:
            from xgboost import XGBClassifier
        except:
            warnings.warn("Xgboost is not imported! Blending will use linear model instead!")
            meta_learner = 'linear'

        # We use Xgboost as default meta-learner
        if self.task_type in CLS_TASKS:
            if meta_learner == 'linear':
                from sklearn.linear_model.logistic import LogisticRegression
                self.meta_learner = LogisticRegression(max_iter=1000)
            elif meta_learner == 'gb':
                from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
                self.meta_learner = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                               n_estimators=250)
            elif meta_learner == 'xgboost':
                from xgboost import XGBClassifier
                self.meta_learner = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=150)
        else:
            if meta_learner == 'linear':
                from sklearn.linear_model import LinearRegression
                self.meta_learner = LinearRegression()
            elif meta_learner == 'xgboost':
                from xgboost import XGBRegressor
                self.meta_learner = XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=70)

    def fit(self, data):
        # Split training data for phase 1 and phase 2
        if self.task_type in CLS_TASKS:
            kf = StratifiedKFold(n_splits=self.kfold)
        else:
            kf = KFold(n_splits=self.kfold)

        # Train basic models using a part of training data
        model_cnt = 0
        feature_p2 = None
        for algo_id in self.stats["include_algorithms"]:
            train_list = self.stats[algo_id]['train_data_list']
            configs = self.stats[algo_id]['configurations']
            for idx in range(len(train_list)):
                X, y = train_list[idx].data
                for _config in configs:
                    for j, (train, test) in enumerate(kf.split(X, y)):
                        x_p1, x_p2, y_p1, _ = X[train], X[test], y[train], y[test]
                        if self.base_model_mask[model_cnt] == 1:
                            estimator = fetch_predict_estimator(self.task_type, _config, x_p1, y_p1)
                            with open(os.path.join(self.output_dir, 'model%d_part%d' % (model_cnt, j)), 'wb') as f:
                                pkl.dump(estimator, f)
                            if self.task_type in CLS_TASKS:
                                pred = estimator.predict_proba(x_p2)
                                n_dim = np.array(pred).shape[1]
                                if n_dim == 2:
                                    # Binary classificaion
                                    n_dim = 1
                                # Initialize training matrix for phase 2
                                if feature_p2 is None:
                                    num_samples = len(train) + len(test)
                                    feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                                if n_dim == 1:
                                    feature_p2[test, model_cnt * n_dim:(model_cnt + 1) * n_dim] = pred[:, 1:2]
                                else:
                                    feature_p2[test, model_cnt * n_dim:(model_cnt + 1) * n_dim] = pred
                            else:
                                pred = estimator.predict(x_p2)
                                n_dim = np.array(pred).shape[1]
                                # Initialize training matrix for phase 2
                                if feature_p2 is None:
                                    num_samples = len(train) + len(test)
                                    feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                                feature_p2[test, model_cnt * n_dim:(model_cnt + 1) * n_dim] = pred
                    model_cnt += 1
        # Train model for stacking using the other part of training data
        self.meta_learner.fit(feature_p2, y)
        return self

    def get_feature(self, data, solvers):
        # Predict the labels via stacking
        feature_p2 = None
        model_cnt = 0
        for algo_id in self.stats["include_algorithms"]:
            train_list = self.stats[algo_id]['train_data_list']
            configs = self.stats[algo_id]['configurations']
            for train_node in train_list:
                test_node = solvers[algo_id].optimizer['fe'].apply(data, train_node)
                for _ in configs:
                    if self.base_model_mask[model_cnt] == 1:
                        for j in range(self.kfold):
                            with open(os.path.join(self.output_dir, 'model%d_part%d' % (model_cnt, j)), 'rb') as f:
                                estimator = pkl.load(f)
                            if self.task_type in CLS_TASKS:
                                pred = estimator.predict_proba(test_node.data[0])
                                n_dim = np.array(pred).shape[1]
                                if n_dim == 2:
                                    n_dim = 1
                                if feature_p2 is None:
                                    num_samples = len(test_node.data[0])
                                    feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                                # Get average predictions
                                if n_dim == 1:
                                    feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] = \
                                        feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] + pred[:,
                                                                                                   1:2] / self.kfold
                                else:
                                    feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] = \
                                        feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] + pred / self.kfold
                            else:
                                pred = estimator.predict(test_node.data[0])
                                n_dim = np.array(pred).shape[1]
                                # Initialize training matrix for phase 2
                                if feature_p2 is None:
                                    num_samples = len(test_node.data[0])
                                    feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                                # Get average predictions
                                feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] = \
                                    feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] + pred / self.kfold
                model_cnt += 1
        return feature_p2

    def predict(self, data, solvers):
        feature_p2 = self.get_feature(data, solvers)
        # Get predictions from meta-learner
        if self.task_type in CLS_TASKS:
            final_pred = self.meta_learner.predict_proba(feature_p2)
        else:
            final_pred = self.meta_learner.predict(feature_p2)
        return final_pred
