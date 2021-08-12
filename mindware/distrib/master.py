import time
from mindware.distrib.distributed_bo import mqSMBO
from mindware.base_estimator import BaseEstimator
from openbox.core.message_queue.sender_messager import SenderMessager
from openbox.core.message_queue.receiver_messager import ReceiverMessager


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
        self.eval_type = self.estimator.evaluation
        self.output_dir = self.estimator.output_dir
        self.optimizer = mqSMBO(self.evaluator, self.config_space, runtime_limit=self.estimator.time_limit,
                                eval_type=self.eval_type, ip=ip, port=port, authkey=authkey,
                                logging_dir=self.output_dir)

    def build_ensemble(self):
        # Step 1: send message to each worker.
        sender_list = list()
        for _worker_id in self.optimizer.workers:
            sender = SenderMessager()
            sender.send_message()
            sender_list.append(sender)

        # Step 2: fetch result from each worker.
        results = [None] * len(sender_list)
        while True:
            for idx, _sender in enumerate(sender_list):
                try:
                    msg = _sender.receive_message()
                except Exception as e:
                    break
                if msg is not None:
                    results[idx] = msg

            if None not in results:
                break

            time.sleep(1.)
        # Calculate parameters in ensemble selection.

    def run(self):
        self.optimizer.run()
        self.build_ensemble()

    def predict(self):
        pass
