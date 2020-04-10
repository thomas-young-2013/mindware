import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ConfigSpace import Configuration


def execute_func(params):
    start_time = time.time()
    evaluator, config = params
    try:
        if isinstance(config, Configuration):
            evaluator(config)
            score = evaluator(config, name='hpo')
        else:
            score = evaluator(config, data_node=config, name='fe')
    except Exception as e:
        score = -np.inf

    time_taken = time.time() - start_time
    return score, time_taken


def wait_tasks_finish(trial_stats):
    all_completed = False
    while not all_completed:
        all_completed = True
        for trial in trial_stats:
            if not trial.done():
                all_completed = False
                time.sleep(0.1)
                break


def parallel_execute(evaluator, param_list, n_worker=1):
    thread_pool = ThreadPoolExecutor(max_workers=n_worker)
    n_configuration = len(param_list)
    batch_size = n_worker
    n_batch = n_configuration // batch_size + (1 if n_configuration % batch_size != 0 else 0)
    evaluation_result = list()

    for i in range(n_batch):
        execution_stats = list()
        for _param in param_list[i * batch_size: (i + 1) * batch_size]:
            execution_stats.append(thread_pool.submit(execute_func,
                                   (evaluator, _param)))
        # wait a batch of trials finish
        wait_tasks_finish(execution_stats)

        # get the evaluation statistics
        for trial in execution_stats:
            assert (trial.done())
            perf = trial.result()
            evaluation_result.append(perf)
    thread_pool.shutdown(wait=True)
    return evaluation_result
