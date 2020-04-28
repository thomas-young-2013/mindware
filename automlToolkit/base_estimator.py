import os
from automlToolkit.automl import AutoML
from automlToolkit.components.metrics.metric import get_metric
from automlToolkit.components.feature_engineering.transformation_graph import DataNode


class BaseEstimator(object):
    def __init__(
            self,
            metric=None,
            time_limit=None,
            iter_num_per_algo=50,
            include_algorithms=None,
            ensemble_method='ensemble_selection',
            ensemble_size=20,
            per_run_time_limit=150,
            random_state=1,
            n_jobs=1,
            evaluation='holdout',
            output_dir="/tmp/"):
        self.metric = metric
        self.task_type = None
        self.time_limit = time_limit
        self.iter_num_per_algo = iter_num_per_algo
        self.include_algorithms = include_algorithms
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.per_run_time_limit = per_run_time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.evaluation = evaluation
        self.output_dir = output_dir
        self._ml_engine = None
        # Create output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def build_engine(self):
        """Build AutoML controller"""
        engine = self.get_automl()(
            task_type=self.task_type,
            metric=self.metric,
            time_limit=self.time_limit,
            iter_num_per_algo=self.iter_num_per_algo,
            include_algorithms=self.include_algorithms,
            ensemble_method=self.ensemble_method,
            ensemble_size=self.ensemble_size,
            per_run_time_limit=self.per_run_time_limit,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            evaluation=self.evaluation,
            output_dir=self.output_dir
        )
        return engine

    def fit(self, data: DataNode):
        assert data is not None and isinstance(data, DataNode)
        self.metric = get_metric(self.metric)
        self._ml_engine = self.build_engine()
        self._ml_engine.fit(data)
        return self

    def predict(self, X: DataNode, batch_size=None, n_jobs=1):
        return self._ml_engine.predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def score(self, data: DataNode):
        raise NotImplementedError()
        # return self._ml_engine.score(data)

    def predict_proba(self, X: DataNode, batch_size=None, n_jobs=1):
        return self._ml_engine.predict_proba(X, batch_size=batch_size, n_jobs=n_jobs)

    def get_automl(self):
        return AutoML

    def show_info(self):
        raise NotImplementedError()
