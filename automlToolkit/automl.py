import time
import typing
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit

classification_algorithms = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
regression_algorithms = ['random_forest']


class AutoML(object):
    def __init__(self, classification=True,
                 time_limit=None,
                 iter_num_per_algo=50,
                 include_algorithms=None,
                 ensemble_size=20,
                 per_run_time_limit=150,
                 random_state=1,
                 n_jobs=1,
                 evaluation='holdout',
                 output_dir=None):
        self.is_classification = classification
        self.time_limit = time_limit
        self.random_seed = random_state
        self.ensemble_size = ensemble_size
        self.per_run_time_limit = per_run_time_limit
        self.iter_num_per_algo = iter_num_per_algo
        self.output_dir = output_dir
        self.evaluation_type = evaluation
        self.n_jobs = n_jobs
        self.solvers = dict()
        if include_algorithms is None:
            if self.is_classification:
                self.include_algorithms = classification_algorithms
            else:
                self.include_algorithms = regression_algorithms

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
            self.solvers[_algo] = SecondLayerBandit(
                _algo, train_data, output_dir=self.output_dir,
                per_run_time_limit=self.per_run_time_limit,
                seed=self.random_seed,
                eval_type=self.evaluation_type,
                dataset_id=dataset_id,
                n_jobs=self.n_jobs,
                mth='alter_hpo'
            )

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
                if self.time_limit is None:
                    if time.time() - _start_time >= time_limit_per_algo:
                        break
                if solver.early_stopped_flag:
                    break

        # Ensembling all intermediate/ultimate models found in above optimization process.
        pass

    def predict(self, test_data: DataNode):
        pass
