import time
import numpy as np
import multiprocessing
from ConfigSpace import Configuration


def execute_func(params):
    start_time = time.time()
    evaluator, config, subsample_ratio = params
    try:
        if isinstance(config, Configuration):
            score = evaluator(config, name='hpo', resource_ratio=subsample_ratio)
        else:
            score = evaluator(None, data_node=config, name='fe', resource_ratio=subsample_ratio)
    except Exception as e:
        score = np.inf

    time_taken = time.time() - start_time
    return score, time_taken


class ParallelProcessEvaluator(object):
    def __init__(self, evaluator, n_worker=1):
        self.evaluator = evaluator
        self.n_worker = n_worker
        self.process_pool = multiprocessing.Pool(processes=self.n_worker)

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

    def parallel_execute(self, param_list, resource_ratio=1.):
        evaluation_result = list()
        apply_results = list()

        for _param in param_list:
            apply_results.append(self.process_pool.apply_async(execute_func,
                                                                 (self.evaluator, _param, resource_ratio)))
        for res in apply_results:
            res.wait()
            perf = res.get()
            evaluation_result.append(perf)

        return evaluation_result
