import time
from concurrent.futures import ThreadPoolExecutor

from solnml.components.evaluators.base_evaluator import fetch_predict_estimator


def execute_func(params):
    estimator = fetch_predict_estimator(*params)
    return estimator


class ParallelFetcher(object):
    def __init__(self, n_worker=1):
        self.n_worker = n_worker
        self.thread_pool = ThreadPoolExecutor(max_workers=n_worker)
        self.execution_stats = list()
        self.estimators = list()

    def wait_tasks_finish(self):
        all_completed = False
        while not all_completed:
            all_completed = True
            for trial in self.execution_stats:
                if not trial.done():
                    all_completed = False
                    time.sleep(0.1)
                    break
        for trial in self.execution_stats:
            assert (trial.done())
            estimator = trial.result()
            self.estimators.append(estimator)
        return self.estimators

    def submit(self, task_type, config, X_train, y_train, weight_balance, data_balance):
        self.execution_stats.append(self.thread_pool.submit(execute_func,
                                                            (task_type, config, X_train, y_train, weight_balance,
                                                             data_balance)))
