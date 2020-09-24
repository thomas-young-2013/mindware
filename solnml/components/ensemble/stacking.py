import numpy as np
import warnings
import os
import pickle as pkl
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics.scorer import _BaseScorer

from solnml.components.ensemble.base_ensemble import BaseEnsembleModel
from solnml.components.utils.constants import CLS_TASKS
from solnml.components.evaluators.base_evaluator import fetch_predict_estimator


class Stacking(BaseEnsembleModel):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None,
                 meta_learner='lightgbm',
                 kfold=5):
        super().__init__(stats=stats,
                         ensemble_method='stacking',
                         ensemble_size=ensemble_size,
                         task_type=task_type,
                         metric=metric,
                         output_dir=output_dir)

        self.kfold = kfold
        try:
            from lightgbm import LGBMClassifier
        except:
            warnings.warn("Lightgbm is not imported! Stacking will use linear model instead!")
            meta_learner = 'linear'

        self.meta_method = meta_learner

        # We use Xgboost as default meta-learner
        if self.task_type in CLS_TASKS:
            if meta_learner == 'linear':
                from sklearn.linear_model.logistic import LogisticRegression
                self.meta_learner = LogisticRegression(max_iter=1000)
            elif meta_learner == 'gb':
                from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
                self.meta_learner = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                               n_estimators=250)
            elif meta_learner == 'lightgbm':
                from lightgbm import LGBMClassifier
                self.meta_learner = LGBMClassifier(max_depth=4, learning_rate=0.05, n_estimators=150)
        else:
            if meta_learner == 'linear':
                from sklearn.linear_model import LinearRegression
                self.meta_learner = LinearRegression()
            elif meta_learner == 'lightgbm':
                from lightgbm import LGBMRegressor
                self.meta_learner = LGBMRegressor(max_depth=4, learning_rate=0.05, n_estimators=70)

    def fit(self, data):
        # Split training data for phase 1 and phase 2
        if self.task_type in CLS_TASKS:
            kf = StratifiedKFold(n_splits=self.kfold)
        else:
            kf = KFold(n_splits=self.kfold)

        # Train basic models using a part of training data
        model_cnt = 0
        suc_cnt = 0
        feature_p2 = None
        for algo_id in self.stats["include_algorithms"]:
            model_to_eval = self.stats[algo_id]['model_to_eval']
            for idx, (node, config) in enumerate(model_to_eval):
                X, y = node.data
                if self.base_model_mask[model_cnt] == 1:
                    for j, (train, test) in enumerate(kf.split(X, y)):
                        x_p1, x_p2, y_p1, _ = X[train], X[test], y[train], y[test]
                        estimator = fetch_predict_estimator(self.task_type, config, x_p1, y_p1,
                                                            weight_balance=data.enable_balance,
                                                            data_balance=data.data_balance
                                                            )
                        with open(
                                os.path.join(self.output_dir, '%s-model%d_part%d' % (self.timestamp, model_cnt, j)),
                                'wb') as f:
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
                                feature_p2[test, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred[:, 1:2]
                            else:
                                feature_p2[test, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                        else:
                            pred = estimator.predict(x_p2).reshape(-1, 1)
                            n_dim = 1
                            # Initialize training matrix for phase 2
                            if feature_p2 is None:
                                num_samples = len(train) + len(test)
                                feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                            feature_p2[test, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                    suc_cnt += 1
                model_cnt += 1
        # Train model for stacking using the other part of training data
        self.meta_learner.fit(feature_p2, y)
        return self

    def get_feature(self, data, solvers):
        # Predict the labels via stacking
        feature_p2 = None
        model_cnt = 0
        suc_cnt = 0
        for algo_id in self.stats["include_algorithms"]:
            model_to_eval = self.stats[algo_id]['model_to_eval']
            for idx, (node, config) in enumerate(model_to_eval):
                test_node = solvers[algo_id].optimizer['fe'].apply(data, node)
                if self.base_model_mask[model_cnt] == 1:
                    for j in range(self.kfold):
                        with open(
                                os.path.join(self.output_dir, '%s-model%d_part%d' % (self.timestamp, model_cnt, j)),
                                'rb') as f:
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
                                feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = \
                                    feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] + pred[:,
                                                                                           1:2] / self.kfold
                            else:
                                feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = \
                                    feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] + pred / self.kfold
                        else:
                            pred = estimator.predict(test_node.data[0]).reshape(-1, 1)
                            n_dim = 1
                            # Initialize training matrix for phase 2
                            if feature_p2 is None:
                                num_samples = len(test_node.data[0])
                                feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                            # Get average predictions
                            feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = \
                                feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] + pred / self.kfold
                    suc_cnt += 1
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

    def get_ens_model_info(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        for algo_id in self.stats["include_algorithms"]:
            model_to_eval = self.stats[algo_id]['model_to_eval']
            for idx, (node, config) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    model_path = os.path.join(self.output_dir, '%s-stacking-model%d' % (self.timestamp, model_cnt))
                    ens_config.append((algo_id, node.config, config, model_path))
                model_cnt += 1
        ens_info['ensemble_method'] = 'stacking'
        ens_info['config'] = ens_config
        ens_info['meta_learner'] = self.meta_method
        ens_info['meta_model'] = self.meta_learner
        return ens_info
