import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from autosklearn.metrics import accuracy

from automlToolkit.components.evaluator import get_estimator
from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.components.ensemble.ensemble_selection import EnsembleSelection
from automlToolkit.components.feature_engineering.transformation_graph import DataNode


class EnsembleBuilder(object):
    def __init__(self, bandit: FirstLayerBandit, ensemble_size=50):
        self.bandit = bandit
        self.ensemble_size = ensemble_size

    def fit_predict(self, test_data: DataNode):
        n_best = 10
        test_size = 0.2
        stats = self.bandit.fetch_ensemble_members(test_data, mode=False)
        seed = stats['split_seed']

        train_predictions = []
        test_predictions = []
        for algo_id in self.bandit.nbest_algo_ids:
            best_configs = stats[algo_id]['configurations']

            train_list, test_list = stats[algo_id]['train_data_list'], stats[algo_id]['test_data_list']
            for idx in range(len(train_list)):
                X, y = train_list[idx].data
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
                for train_index, test_index in sss.split(X, y):
                    X_train, X_valid = X[train_index], X[test_index]
                    y_train, y_valid = y[train_index], y[test_index]
                X_test, y_test = test_list[idx].data

                for config in best_configs:
                    # Build the ML estimator.
                    _, estimator = get_estimator(config)
                    estimator.fit(X_train, y_train)
                    y_pred = estimator.predict_proba(X_valid)
                    train_predictions.append(y_pred)
                    y_pred = estimator.predict_proba(X_test)
                    test_predictions.append(y_pred)

        es = EnsembleSelection(ensemble_size=self.ensemble_size,
                               task_type=1, metric=accuracy,
                               random_state=np.random.RandomState(seed))
        es.fit(train_predictions, y_valid, identifiers=None)
        y_pred = es.predict(test_predictions)
        y_pred = np.argmax(y_pred, axis=-1)
        return y_pred

    def score(self, test_data: DataNode):
        y_pred = self.fit_predict(test_data)
        return accuracy_score(test_data.data[1], y_pred)
