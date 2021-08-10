import time

from mindware.distrib.distributed_bo import mqSMBO
from mindware.base_estimator import BaseEstimator


class Master(object):
    """
        Master.
            the master adopts a specific optimization strategy to conduct configuration search.
    """

    def __init__(self, estimator: BaseEstimator, optimize_method='bo', ip="127.0.0.1", port=13579, authkey=b'abc'):
        self.estimator = estimator
        self.optimize_method = optimize_method
        self.ip = ip
        self.port = port
        self.authkey = authkey

        self.evaluator = self.estimator.get_evaluator()
        self.config_space = self.estimator.get_config_space()
        self.optimizer = mqSMBO(self.evaluator, self.config_space, runtime_limit=self.estimator.time_limit,
                                ip=ip, port=port, authkey=authkey)

    def run(self):
        self.optimizer.run()
