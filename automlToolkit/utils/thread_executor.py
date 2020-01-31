import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from multiprocessing import Queue
from typing import Callable
from typing import Iterable
from typing import Dict
from typing import Any


class ProcessKillingExecutor:
    """
    The ProcessKillingExecutor works like an `Executor
    <https://docs.python.org/dev/library/concurrent.futures.html#executor-objects>`_
    in that it uses a bunch of processes to execute calls to a function with
    different arguments asynchronously.

    But other than the `ProcessPoolExecutor
    <https://docs.python.org/dev/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>`_,
    the ProcessKillingExecutor forks a new Process for each function call that
    terminates after the function returns or if a timeout occurs.

    This means that contrary to the Executors and similar classes provided by
    the Python Standard Library, you can rely on the fact that a process will
    get killed if a timeout occurs and that absolutely no side can occur
    between function calls.

    Note that descendant processes of each process will not be terminated –
    they will simply become orphaned.
    """

    def __init__(self, max_workers: int=None):
        self.processes = max_workers or os.cpu_count()

    def map(self,
            func: Callable,
            iterable: Iterable,
            timeout: float=None,
            callback_timeout: Callable=None,
            daemon: bool = True
            ) -> Iterable:
        """
        :param func: the function to execute
        :param iterable: an iterable of function arguments
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called, if the task
                times out. It gets the same arguments as the original function
        :param daemon: define the child process as daemon
        """
        executor = ThreadPoolExecutor(max_workers=self.processes)
        params = ({'func': func, 'fn_args': p_args, "p_kwargs": {},
                   'timeout': timeout, 'callback_timeout': callback_timeout,
                   'daemon': daemon} for p_args in iterable)
        return executor.map(self._submit_unpack_kwargs, params)

    def _submit_unpack_kwargs(self, params):
        """ unpack the kwargs and call submit """

        return self.submit(**params)

    def submit(self,
               func: Callable,
               fn_args: Any,
               p_kwargs: Dict,
               timeout: float,
               callback_timeout: Callable[[Any], Any],
               daemon: bool):
        """
        Submits a callable to be executed with the given arguments.
        Schedules the callable to be executed as func(*args, **kwargs) in a new
         process.
        :param func: the function to execute
        :param fn_args: the arguments to pass to the function. Can be one argument
                or a tuple of multiple args.
        :param p_kwargs: the kwargs to pass to the function
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called with the same
                arguments, if the task times out.
        :param daemon: run the child process as daemon
        :return: the result of the function, or None if the process failed or
                timed out
        """
        p_args = fn_args if isinstance(fn_args, tuple) else (fn_args,)
        queue = Queue()
        p = Process(target=self._process_run,
                    args=(queue, func, fn_args,), kwargs=p_kwargs)

        if daemon:
            p.deamon = True

        p.start()
        p.join(timeout=timeout)
        if not queue.empty():
            return queue.get()
        if callback_timeout:
            callback_timeout(*p_args, **p_kwargs)
        if p.is_alive():
            p.terminate()
            p.join()

    @staticmethod
    def _process_run(queue: Queue, func: Callable[[Any], Any]=None,
                     *args, **kwargs):
        """
        Executes the specified function as func(*args, **kwargs).
        The result will be stored in the shared dictionary
        :param func: the function to execute
        :param queue: a Queue
        """
        queue.put(func(*args, **kwargs))
