import time
import os
import numpy as np
import pickle as pkl
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from automlToolkit.components.metrics.metric import get_metric
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.components.utils.constants import CLS_TASKS, REG_TASKS
from automlToolkit.components.ensemble.ensemble_selection import EnsembleSelection
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.evaluators.base_evaluator import fetch_predict_estimator

# TODO: this default value should be updated.
classification_algorithms = ['liblinear_svc', 'random_forest']
regression_algorithms = ['liblinear_svr', 'random_forest']


class AutoML(object):
    def __init__(self,
                 task_type=None,
                 metric='acc',
                 time_limit=None,
                 iter_num_per_algo=50,
                 include_algorithms=None,
                 ensemble_size=20,
                 per_run_time_limit=150,
                 random_state=1,
                 n_jobs=1,
                 evaluation='holdout',
                 output_dir="/tmp/"):
        self.model_cnt = 0
        self.metric = get_metric(metric)
        self.time_limit = time_limit
        self.seed = random_state
        self.ensemble_size = ensemble_size
        self.per_run_time_limit = per_run_time_limit
        self.iter_num_per_algo = iter_num_per_algo
        self.output_dir = output_dir
        self.evaluation_type = evaluation
        self.n_jobs = n_jobs
        self.solvers = dict()
        self.task_type = task_type
        if task_type in CLS_TASKS:
            self.include_algorithms = classification_algorithms
        elif task_type in REG_TASKS:
            self.include_algorithms = regression_algorithms
        else:
            raise ValueError("Unknown task type %s" % task_type)

    def fetch_ensemble_members(self, threshold=0.85):
        stats = dict()
        stats['split_seed'] = self.seed
        best_perf = float('-INF')
        for algo_id in self.include_algorithms:
            best_perf = max(best_perf, self.solvers[algo_id].incumbent_perf)
        for algo_id in self.include_algorithms:
            data = dict()
            inc = self.solvers[algo_id].inc
            local_inc = self.solvers[algo_id].local_inc
            fe_optimizer = self.solvers[algo_id].optimizer['fe']
            hpo_optimizer = self.solvers[algo_id].optimizer['hpo']

            train_data_candidates = [inc['fe'], local_inc['fe'], self.solvers[algo_id].original_data]
            for _feature_set in fe_optimizer.features_hist:
                if _feature_set not in train_data_candidates:
                    train_data_candidates.append(_feature_set)

            train_data_list = list()
            for item in train_data_candidates:
                if item not in train_data_list:
                    train_data_list.append(item)

            data['train_data_list'] = train_data_list
            print(algo_id, len(train_data_list))

            configs = hpo_optimizer.configs
            perfs = hpo_optimizer.perfs
            best_configs = [self.solvers[algo_id].default_config, inc['hpo'], local_inc['hpo']]
            best_configs = list(set(best_configs))
            threshold = best_perf * threshold
            for idx in np.argsort(-np.array(perfs)):
                if perfs[idx] >= threshold and configs[idx] not in best_configs:
                    best_configs.append(configs[idx])
                if len(best_configs) >= self.ensemble_size / len(self.include_algorithms):
                    break
            data['configurations'] = best_configs

            stats[algo_id] = data
        return stats

    def fit(self, train_data: DataNode, dataset_id=None):
        """
        this function includes this following two procedures.
            1. tune each algorithm's hyperparameters.
            2. engineer each algorithm's features automatically.
        :param train_data:
        :return:
        """
        # Initialize each algorithm's solver.
        for _algo in self.include_algorithms:
            self.solvers[_algo] = SecondLayerBandit(self.task_type, _algo, train_data,
                                                    metric=self.metric,
                                                    output_dir=self.output_dir,
                                                    per_run_time_limit=self.per_run_time_limit,
                                                    seed=self.seed,
                                                    eval_type=self.evaluation_type,
                                                    dataset_id=dataset_id,
                                                    n_jobs=self.n_jobs,
                                                    mth='alter_hpo')

        # Set the resource limit.
        if self.time_limit is not None:
            time_limit_per_algo = self.time_limit / len(self.include_algorithms)
            max_iter_num = 999999
        else:
            time_limit_per_algo = None
            max_iter_num = self.iter_num_per_algo

        # Optimize each algorithm with corresponding solver.
        for algo in self.include_algorithms:
            _start_time, _iter_id = time.time(), 0
            solver = self.solvers[algo]

            while _iter_id < max_iter_num:
                result = solver.play_once()
                print('optimize %s in %d-th iteration: %.3f' % (algo, _iter_id, result))
                _iter_id += 1
                if self.time_limit is not None:
                    if time.time() - _start_time >= time_limit_per_algo:
                        break
                if solver.early_stopped_flag:
                    break

        # TODO: Single model
        if self.ensemble_size > 0:
            self.stats = self.fetch_ensemble_members()
            # Ensembling all intermediate/ultimate models found in above optimization process.
            # TODO: version1.0, support multiple ensemble methods.
            train_predictions = []
            config_list = []
            train_data_dict = {}
            train_labels = None
            seed = self.stats['split_seed']
            for algo_id in self.include_algorithms:
                train_list = self.stats[algo_id]['train_data_list']
                configs = self.stats[algo_id]['configurations']
                for idx in range(len(train_list)):
                    X, y = train_list[idx].data

                    # TODO: Hyperparameter
                    test_size = 0.33

                    if self.task_type in CLS_TASKS:
                        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                    else:
                        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

                    for train_index, test_index in ss.split(X, y):
                        X_train, X_valid = X[train_index], X[test_index]
                        y_train, y_valid = y[train_index], y[test_index]

                    if train_labels is not None:
                        assert (train_labels == y_valid).all()
                    else:
                        train_labels = y_valid
                    for _config in configs:
                        config_list.append(_config)
                        train_data_dict[self.model_cnt] = (X, y)
                        estimator = fetch_predict_estimator(self.task_type, _config, X_train, y_train)
                        with open(os.path.join(self.output_dir, 'model%d' % self.model_cnt), 'wb') as f:
                            pkl.dump(estimator, f)
                        if self.task_type in CLS_TASKS:
                            y_valid_pred = estimator.predict_proba(X_valid)
                        else:
                            y_valid_pred = estimator.predict(X_valid)
                        train_predictions.append(y_valid_pred)
                        self.model_cnt += 1

            self.es = EnsembleSelection(ensemble_size=self.ensemble_size,
                                        task_type=self.task_type, metric=self.metric,
                                        random_state=np.random.RandomState(seed))
            self.es.fit(train_predictions, train_labels, identifiers=None)

    def _predict(self, test_data: DataNode, batch_size=None, n_jobs=1):
        # TODO: Single model
        if self.ensemble_size > 0:
            if self.es is None:
                raise AttributeError("AutoML is not fitted!")

            test_prediction = []
            cur_idx = 0
            for algo_id in self.include_algorithms:
                for train_node in self.stats[algo_id]['train_data_list']:
                    test_node = self.solvers[algo_id].optimizer['fe'].apply(test_data, train_node)
                    X_test, _ = test_node.data
                    for _ in self.stats[algo_id]['configurations']:
                        with open(os.path.join(self.output_dir, 'model%d' % cur_idx), 'rb') as f:
                            estimator = pkl.load(f)
                            if self.task_type in CLS_TASKS:
                                test_prediction.append(estimator.predict_proba(X_test))
                            else:
                                test_prediction.append(estimator.predict(X_test))
                        cur_idx += 1
            return self.es.predict(test_prediction)

    def predict_proba(self, test_data: DataNode, batch_size=None, n_jobs=1):
        if self.task_type in REG_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        return self._predict(test_data, batch_size=batch_size)

    def predict(self, test_data: DataNode, batch_size=None, n_jobs=1):
        if self.task_type in CLS_TASKS:
            pred = self._predict(test_data)
            return np.argmax(pred, axis=-1)
        else:
            return self._predict(test_data, batch_size=batch_size)
