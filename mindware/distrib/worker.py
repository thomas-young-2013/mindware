import sys
import time
import traceback
import numpy as np
import pickle as pkl
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from openbox.core.base import Observation
from openbox.utils.util_funcs import get_result
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.core.message_queue.worker_messager import WorkerMessager
from openbox.core.message_queue.receiver_messager import ReceiverMessager

from mindware.utils.logging_utils import get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.feature_engineering.parse import construct_node
from mindware.distrib.utils import get_host_ip


class BaseWorker(object):
    def __init__(self, estimator, master_ip, master_port, authkey, worker_port):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.estimator = estimator
        self.evaluator = estimator.get_evaluator()
        self.master_ip = master_ip
        self.master_port = master_port
        self.worker_port = worker_port
        self.worker_messager = WorkerMessager(master_ip, master_port, authkey)
        self.receiver_messager = ReceiverMessager(ip='127.0.0.1', port=worker_port)


class EvaluationWorker(BaseWorker):
    def __init__(self, evaluator, master_ip="127.0.0.1", master_port=13579, authkey=b'abc', worker_port=12345):
        super().__init__(evaluator, master_ip, master_port, authkey, worker_port)

        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = float("-INF")
        self.incumbent_config = None
        self.eval_dict = dict()
        self.best_configs = list()
        self.worker_port = worker_port

    def run(self):
        while True:
            try:
                msg = self.receiver_messager.receive_message()
            except Exception as e:
                self.logger.error("Worker receive message error: %s." % str(e))
            if msg == 'ready':
                break

            # Get config
            try:
                msg = self.worker_messager.receive_message()
            except Exception as e:
                self.logger.error("Worker receive message error: %s." % str(e))

            if msg is None:
                # Wait for configs
                time.sleep(0.3)
                continue
            self.logger.info("Worker: get config. start working.")
            config, time_limit_per_trial = msg

            # Start working
            trial_state = SUCCESS
            start_time = time.time()
            try:
                args, kwargs = (config,), dict()
                timeout_status, _result = time_limit(self.evaluator,
                                                     time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
                else:
                    objs, constraints = get_result(_result)
            except Exception as e:
                if isinstance(e, TimeoutException):
                    trial_state = TIMEOUT
                else:
                    traceback.print_exc(file=sys.stdout)
                    trial_state = FAILED
                objs = None
                constraints = None

            _perf = float("INF") if objs is None else objs[0]
            self.configs.append(config)
            self.perfs.append(_perf)
            self.eval_dict[config] = [-_perf, time.time(), trial_state]

            if -_perf > self.incumbent_perf:
                self.incumbent_perf = -_perf
                self.incumbent_config = config

            elapsed_time = time.time() - start_time
            observation = Observation(config, trial_state, constraints, objs, elapsed_time,
                                      worker_info={'ip': get_host_ip(), 'port': self.worker_port})

            # Send result
            self.logger.info("Worker: observation=%s. sending result." % str(observation))
            try:
                self.worker_messager.send_message(observation)
            except Exception as e:
                self.logger.error("Worker send message error: %s." % str(e))

        eval_list = self.eval_dict.items()
        sorted_list = sorted(eval_list, key=lambda x: x[1][0], reverse=True)
        if len(sorted_list) > 10:
            ensemble_dict = dict(sorted_list[:int(len(sorted_list) / 10)])
        else:
            ensemble_dict = dict(sorted_list[:1])
        self.best_configs = list(ensemble_dict.keys())
        preds = self.fetch_ensemble_pred()
        self.receiver_messager.send_message(preds)

        while True:
            time.sleep(5)
            try:
                msg = self.receiver_messager.receive_message()
            except Exception as e:
                self.logger.error("Worker receive message error: %s." % str(e))
            if msg == 'over':
                break
            elif msg is not None:
                self.receiver_messager.send_message(msg)

    def fetch_ensemble_pred(self):
        if self.evaluator.resampling_params is None or 'test_size' not in self.evaluator.resampling_params:
            test_size = 0.33
        else:
            test_size = self.evaluator.resampling_params['test_size']

        preds = list()

        for config in self.best_configs:
            # Convert Configuration into dictionary
            if not isinstance(config, dict):
                config = config.get_dictionary().copy()
            else:
                config = config.copy()

            model_path = CombinedTopKModelSaver.get_path_by_config(output_dir=self.evaluator.output_dir,
                                                                   config=config,
                                                                   identifier=self.evaluator.timestamp)

            with open(model_path, 'rb') as f:
                op_list, model, _ = pkl.load(f)

            _node = self.evaluator.data_node.copy_()
            node = construct_node(_node, op_list)

            if self.estimator.task_type in CLS_TASKS:
                ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.evaluator.seed)
            else:
                ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=self.evaluator.seed)

            for train_index, test_index in ss.split(node.data[0], node.data[1]):
                _x_val = node.data[0][test_index]

            if self.estimator.task_type in CLS_TASKS:
                pred = model.predict_proba(_x_val)
            else:
                pred = model.predict(_x_val)
            preds.append(pred)

        preds = np.array(preds)
        return preds

    def predict(self, datanode):
        preds = list()

        for config in self.best_configs:
            # Convert Configuration into dictionary
            if not isinstance(config, dict):
                config = config.get_dictionary().copy()
            else:
                config = config.copy()

            model_path = CombinedTopKModelSaver.get_path_by_config(output_dir=self.evaluator.output_dir,
                                                                   config=config,
                                                                   identifier=self.evaluator.timestamp)

            with open(model_path, 'rb') as f:
                op_list, model, _ = pkl.load(f)

            _node = datanode.copy_()
            node = construct_node(_node, op_list)
            x_test = node.data[0]

            if self.estimator.task_type in CLS_TASKS:
                pred = model.predict_proba(x_test)
            else:
                pred = model.predict(x_test)
            preds.append(pred)

        preds = np.array(preds)

        while True:
            try:
                msg = self.receiver_messager.receive_message()
            except Exception as e:
                self.logger.error("Worker receive message error: %s." % str(e))
            if msg == 'ready':
                break
        self.receiver_messager.send_message(preds)

        while True:
            time.sleep(5)
            try:
                msg = self.receiver_messager.receive_message()
            except Exception as e:
                self.logger.error("Worker receive message error: %s." % str(e))
            if msg == 'over':
                break
            elif msg is not None:
                self.receiver_messager.send_message(msg)
