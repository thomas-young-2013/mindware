import time
import numpy as np
from ConfigSpace import Configuration
from .base.nondaemonic_processpool import ProcessPool


def execute_func(evaluator, config, subsample_ratio):
    start_time = time.time()
    try:
        if isinstance(config, Configuration):
            score = evaluator(config, name='hpo', resource_ratio=subsample_ratio)
        else:
            score = evaluator(None, data_node=config, name='fe', resource_ratio=subsample_ratio)
    except Exception as e:
        print(e)
        score = np.inf

    time_taken = time.time() - start_time
    return score, time_taken


class ParallelProcessEvaluator(object):
    def __init__(self, evaluator, n_worker=1):
        self.evaluator = evaluator
        self.n_worker = n_worker
        self.process_pool = None

    def update_evaluator(self, evaluator):
        self.evaluator = evaluator

    def parallel_execute(self, param_list, resource_ratio=1.):
        evaluation_result = list()
        apply_results = list()

        for _param in param_list:
            apply_results.append(self.process_pool.apply_async(execute_func,
                                                               (self.evaluator, _param, resource_ratio)))
        for res in apply_results:
            res.wait()
            perf = res.get()[0]
            evaluation_result.append(perf)

        return evaluation_result

    def shutdown(self):
        self.process_pool.close()

    def __enter__(self):
        self.process_pool = ProcessPool(processes=self.n_worker)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process_pool.close()
