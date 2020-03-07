import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from autosklearn.metrics import balanced_accuracy

from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.components.ensemble.ensemble_selection import EnsembleSelection
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.evaluators.evaluator import fetch_predict_estimator


class EnsembleBuilder(object):
    def __init__(self, bandit: FirstLayerBandit, ensemble_size=50, n_jobs=1):
        self.bandit = bandit
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs

    def fit_predict(self, test_data: DataNode):
        test_size = 0.33
        stats = self.bandit.fetch_ensemble_members(test_data, mode=False)
        seed = stats['split_seed']

        print('Start to train base models from %d algorithms!' % len(self.bandit.nbest_algo_ids))
        start_time = time.time()

        train_predictions = []
        train_labels = None
        test_predictions = []

        def evaluate(_config, train_X, train_y, valid_X, test_X):
            # Build the ML estimator.
            estimator = fetch_predict_estimator(_config, train_X, train_y)
            y_valid_pred = estimator.predict_proba(valid_X)
            y_test_pred = estimator.predict_proba(test_X)
            return y_valid_pred, y_test_pred

        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            for algo_id in self.bandit.nbest_algo_ids:
                best_configs = stats[algo_id]['configurations']
                train_list, test_list = stats[algo_id]['train_data_list'], stats[algo_id]['test_data_list']
                print(algo_id, len(best_configs) * len(train_list))

                for idx in range(len(train_list)):
                    X, y = train_list[idx].data
                    X_test, y_test = test_list[idx].data

                    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
                    for train_index, test_index in sss.split(X, y):
                        X_train, X_valid = X[train_index], X[test_index]
                        y_train, y_valid = y[train_index], y[test_index]

                    if train_labels is not None:
                        assert (train_labels == y_valid).all()
                    else:
                        train_labels = y_valid

                    task_list = []
                    for config in best_configs:
                        task_list.append(pool.submit(evaluate, config, X_train, y_train, X_valid, X_test))

                    for task in as_completed(task_list):
                        try:
                            y_valid_pred, y_test_pred = task.result()
                            train_predictions.append(y_valid_pred)
                            test_predictions.append(y_test_pred)
                        except Exception as e:
                            print(e)

        print('Training Base models ends!')
        print('It took %.2f seconds!' % (time.time() - start_time))

        es = EnsembleSelection(ensemble_size=self.ensemble_size,
                               task_type=1, metric=balanced_accuracy,
                               random_state=np.random.RandomState(seed))
        es.fit(train_predictions, train_labels, identifiers=None)
        y_pred = es.predict(test_predictions)
        y_pred = np.argmax(y_pred, axis=-1)
        return y_pred

    def score(self, test_data: DataNode, metric_func=None):
        if metric_func is None:
            metric_func = accuracy_score
        y_pred = self.fit_predict(test_data)
        return metric_func(test_data.data[1], y_pred)
