import time
import signal
import timeout_decorator
from contextlib import contextmanager


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def long_function_call():
    a = 1
    for _ in range(10000000):
        a += 10
        if a > 10000:
            a = 1
        print(a)
    print(a)


def test_simple_func():
    try:
        with time_limit(10):
            long_function_call()
    except TimeoutError as e:
        print("Timed out!")


@timeout_decorator.timeout(5, timeout_exception=TimeoutError, use_signals=False)
def test():
    print("Start")
    for i in range(1, 10):
        time.sleep(1)
        print("{} seconds have passed".format(i))


test()
