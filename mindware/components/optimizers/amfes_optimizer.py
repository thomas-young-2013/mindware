import time
import traceback
from ConfigSpace import ConfigurationSpace
from tuner.async_mq_mfes import async_mqMFES
from mindware.components.optimizers.base_optimizer import BaseOptimizer


class AsyncMFES(async_mqMFES, BaseOptimizer):
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 name, timestamp, eval_type,
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
                 random_state=1,
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',
                 output_dir=None):
        async_mqMFES.__init__(self, objective_func=objective_func,
                              config_space=config_space,
                              R=R, eta=eta,
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
                              random_state=random_state,
                              restart_needed=restart_needed,
                              time_limit_per_trial=time_limit_per_trial,
                              runtime_limit=runtime_limit,
                              ip=ip, port=port, authkey=authkey)

        BaseOptimizer.__init__(self, evaluator=objective_func,
                               config_space=config_space,
                               name=name, timestamp=timestamp,
                               eval_type=eval_type,
                               output_dir=output_dir, seed=random_state)

    def run(self):
        try:
            worker_num = 0
            while True:
                if self.runtime_limit is not None and time.time() - self.global_start_time > self.runtime_limit:
                    self.logger.info('RUNTIME BUDGET is RUNNING OUT.')
                    return

                # Get observation from worker
                observation = self.master_messager.receive_message()  # return_info, time_taken, trial_id, config
                if observation is None:
                    # Wait for workers.
                    time.sleep(self.sleep_time)
                    continue

                return_info, time_taken, trial_id, config = observation
                # worker init
                if config is None:
                    worker_num += 1
                    self.logger.info("Worker %d init." % (worker_num,))
                # update observation
                else:
                    global_time = time.time() - self.global_start_time
                    self.logger.info('Master get observation: %s. Global time=%.2fs.' % (str(observation), global_time))
                    n_iteration = return_info['n_iteration']
                    perf = return_info['loss']
                    t = time.time()
                    self.update_observation(config, perf, n_iteration)
                    self.logger.info('update_observation() cost %.2fs.' % (time.time() - t,))
                    self.recorder.append({'trial_id': trial_id, 'time_consumed': time_taken,
                                          'configuration': config, 'n_iteration': n_iteration,
                                          'return_info': return_info, 'global_time': global_time})

                    if (not hasattr(self, 'R')) or n_iteration == self.R:
                        self.save_intermediate_statistics()
                        self.update_saver([config], [perf])

                # Send new job
                t = time.time()
                config, n_iteration, extra_conf = self.get_job()
                self.logger.info('get_job() cost %.2fs.' % (time.time() - t,))
                kwargs = {
                    'resource_ratio': float(n_iteration / self.R),
                    'eta': self.eta,
                    'first_iter': (n_iteration == 1)
                }
                msg = [config, kwargs, self.time_limit_per_trial, n_iteration, self.global_trial_counter]
                self.master_messager.send_message(msg)
                self.global_trial_counter += 1
                self.logger.info('Master send job: %s.' % (msg,))

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.logger.error(traceback.format_exc())
