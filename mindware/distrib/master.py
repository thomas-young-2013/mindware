import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from openbox.core.message_queue.sender_messager import SenderMessager

from mindware.distrib.distributed_bo import mqSMBO
from mindware.distrib.ensemble_util import EnsembleSelection
from mindware.base_estimator import BaseEstimator
from mindware.components.utils.constants import CLS_TASKS


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
        self.ensemble = EnsembleSelection(self.estimator.ensemble_size,
                                          self.estimator.task_type,
                                          self.evaluator.scorer)

    def build_ensemble(self):
        # Step 1: send message to each worker.
        sender_list = list()
        for _worker_id in self.optimizer.workers:
            # TODO:
            worker_ip = self.optimizer.workers[_worker_id]['ip']
            worker_port = self.optimizer.workers[_worker_id]['port']
            sender = SenderMessager(ip='127.0.0.1', port=worker_port)
            sender.send_message('ready')
            sender_list.append(sender)

        # Step 2: fetch result from each worker.
        results = [None] * len(sender_list)
        while True:
            for idx, _sender in enumerate(sender_list):
                if results[idx] is not None:
                    continue
                try:
                    msg = _sender.receive_message()
                except Exception as e:
                    print(e)
                    break
                if msg == 'ready':
                    _sender.send_message(msg)
                elif msg is not None:
                    results[idx] = msg
                    _sender.send_message('over')

            all_ready = True
            for result in results:
                if result is None:
                    all_ready = False
                    break
            if all_ready:
                break
            time.sleep(5)

        # Calculate parameters in ensemble selection.
        all_preds = np.vstack(results)

        if self.evaluator.resampling_params is None or 'test_size' not in self.evaluator.resampling_params:
            test_size = 0.33
        else:
            test_size = self.evaluator.resampling_params['test_size']

        if self.estimator.task_type in CLS_TASKS:
            ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.evaluator.seed)
        else:
            ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=self.evaluator.seed)

        node = self.evaluator.data_node.copy_()
        for train_index, test_index in ss.split(node.data[0], node.data[1]):
            _y_val = node.data[1][test_index]
        self.ensemble.fit(all_preds, _y_val)

    def run(self):
        self.optimizer.run()
        self.build_ensemble()

    def _predict(self):
        # Step 1: send message to each worker.
        sender_list = list()
        for _worker_id in self.optimizer.workers:
            # TODO:
            worker_ip = self.optimizer.workers[_worker_id]['ip']
            worker_port = self.optimizer.workers[_worker_id]['port']
            sender = SenderMessager(ip='127.0.0.1', port=worker_port)
            sender.send_message('ready')
            sender_list.append(sender)

        # Step 2: fetch result from each worker.
        results = [None] * len(sender_list)
        while True:
            time.sleep(5)
            for idx, _sender in enumerate(sender_list):
                if results[idx] is not None:
                    continue
                try:
                    msg = _sender.receive_message()
                except Exception as e:
                    break
                if msg == 'ready':
                    _sender.send_message(msg)
                elif msg is not None:
                    results[idx] = msg
                    _sender.send_message('over')

            all_ready = True
            for result in results:
                if result is None:
                    all_ready = False
                    break
            if all_ready:
                break
            time.sleep(5)

        all_preds = np.vstack(results)

        return self.ensemble.predict(all_preds)

    def predict_proba(self):
        if self.estimator.task_type not in CLS_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        return self._predict()

    def predict(self):
        if self.estimator.task_type in CLS_TASKS:
            pred = self._predict()
            return np.argmax(pred, axis=-1)
        else:
            return self._predict()
