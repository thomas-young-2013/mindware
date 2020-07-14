import os
import sys
import time

sys.path.append(os.getcwd())
from solnml.utils.proc_thread.proc_func import kill_proc_tree
from solnml.components.computation.parallel_process import ParallelProcessEvaluator


def evaluate_func(a=None, data_node=None, name='fe', resource_ratio=0.1):
    cnt = 0
    for i in range(100000000):
        cnt += 129
        cnt %= 123200
        for _ in range(100):
            cnt += 3
    return 2 * data_node


try:
    executor = ParallelProcessEvaluator(evaluate_func, n_worker=3)
    _configs = [1, 2, 3, 4, 5, 6]
    res = executor.parallel_execute(_configs, resource_ratio=0.1)
    print(res)
except Exception as e:
    print(e)
finally:
    pid = os.getpid()
    kill_proc_tree(pid, including_parent=False)

print('Task finished.')
print('=' * 100)
