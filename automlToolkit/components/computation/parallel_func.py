import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ConfigSpace import Configuration


def execute_func(params):
    start_time = time.time()
    evaluator, config, subsample_ratio = params
    try:
        if isinstance(config, Configuration):
            score = evaluator(config, name='hpo', data_subsample_ratio=subsample_ratio)
        else:
            score = evaluator(None, data_node=config, name='fe', data_subsample_ratio=subsample_ratio)
    except Exception as e:
        score = -np.inf

    time_taken = time.time() - start_time
    return score, time_taken


class ParallelExecutor(object):
    def __init__(self, evaluator, n_worker=1):
        self.evaluator = evaluator
        self.n_worker = n_worker
        self.thread_pool = ThreadPoolExecutor(max_workers=n_worker)

    def update_evaluator(self, evaluator):
        self.evaluator = evaluator

    def wait_tasks_finish(self, trial_stats):
        all_completed = False
        while not all_completed:
            all_completed = True
            for trial in trial_stats:
                if not trial.done():
                    all_completed = False
                    time.sleep(0.1)
                    break

    def parallel_execute(self, param_list, subsample_ratio=1.):
        n_configuration = len(param_list)
        batch_size = self.n_worker
        n_batch = n_configuration // batch_size + (1 if n_configuration % batch_size != 0 else 0)
        evaluation_result = list()

        for i in range(n_batch):
            execution_stats = list()
            for _param in param_list[i * batch_size: (i + 1) * batch_size]:
                execution_stats.append(self.thread_pool.submit(execute_func,
                                       (self.evaluator, _param, subsample_ratio)))
            # wait a batch of trials finish
            self.wait_tasks_finish(execution_stats)

            # get the evaluation statistics
            for trial in execution_stats:
                assert (trial.done())
                perf = trial.result()
                evaluation_result.append(perf)
        return evaluation_result
