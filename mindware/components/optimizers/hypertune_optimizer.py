import time
import numpy as np
from openbox.utils.config_space import ConfigurationSpace
from multiprocessing import Process, Manager

from mindware.components.optimizers.amfes_optimizer import AsyncMFES
from mindware.components.utils.worker import async_mqmfWorker as Worker


def master_run(return_list, optimizer):
    optimizer.run()
    try:
        optimizer.logger.info('===== bracket status: %s' % optimizer.get_bracket_status(optimizer.bracket))
    except Exception as e:
        pass
    try:
        optimizer.logger.info('===== brackets status: %s' % optimizer.get_brackets_status(optimizer.brackets))
    except Exception as e:
        pass
    return_list.append(optimizer.history_container)  # send to return list


def worker_run(objective_function, ip, port):
    worker = Worker(objective_function, ip, port)
    worker.run()
    print("Worker with port %d exit." % port)


class LocalParallelAMFES:
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 name, timestamp, eval_type,
                 per_run_time_limit=600,
                 inner_iter_num_per_iter=1,
                 R=27,
                 eta=3,
                 skip_outer_loop=0,
                 rand_prob=0.3,
                 init_weight=None, update_enable=True,
                 weight_method='rank_loss_p_norm',
                 fusion_method='idp',
                 power_num=3,
                 surrogate_type='prf',  # 'prf', 'gp'
                 acq_optimizer='local_random',  # 'local_random', 'random'
                 use_weight_init=True,
                 weight_init_choosing='proportional',  # 'proportional', 'pow', 'argmax', 'argmax2'
                 median_imputation=None,  # None, 'top', 'corresponding', 'all'
                 seed=1,
                 restart_needed=True,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',
                 output_dir=None,
                 n_jobs=2):
        self.optimizer = AsyncMFES(objective_func=objective_func,
                                   config_space=config_space,
                                   name=name, timestamp=timestamp,
                                   eval_type=eval_type, R=R, eta=eta,
                                   skip_outer_loop=skip_outer_loop,
                                   rand_prob=rand_prob,
                                   init_weight=init_weight,
                                   update_enable=update_enable,
                                   weight_method=weight_method,
                                   fusion_method=fusion_method,
                                   power_num=power_num,
                                   surrogate_type=surrogate_type,
                                   acq_optimizer=acq_optimizer,
                                   use_weight_init=use_weight_init,
                                   weight_init_choosing=weight_init_choosing,
                                   median_imputation=median_imputation,
                                   random_state=seed,
                                   restart_needed=restart_needed,
                                   time_limit_per_trial=per_run_time_limit,
                                   runtime_limit=runtime_limit,
                                   output_dir=output_dir,
                                   ip=ip, port=port, authkey=authkey)

        self.port = port
        self.ip = ip
        self.authkey = authkey
        self.objective_func = objective_func
        self.n_workers = n_jobs

        self.incumbent_config = None
        self.incumbent_perf = None
        self.recorder = None

    def run(self):
        # Local Parallel
        manager = Manager()
        recorder = manager.list()  # shared list
        master = Process(target=master_run, args=(recorder, self.optimizer))
        master.start()

        time.sleep(5)  # wait for master init
        worker_pool = []
        for i in range(self.n_workers):
            worker = Process(target=worker_run, args=(self.objective_func, '127.0.0.1', self.port))
            worker_pool.append(worker)
            worker.start()

        master.join()  # wait for master to gen result
        for w in worker_pool:
            w.terminate()
            w.join()

        self.recorder = recorder[0]

        return recorder[0]  # covert to list
