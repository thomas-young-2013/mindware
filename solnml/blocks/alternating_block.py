import os
import time
import numpy as np
from ConfigSpace import ConfigurationSpace
from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.components.utils.constants import CLS_TASKS
from solnml.components.utils.topk_saver import CombinedTopKModelSaver
from solnml.blocks.abstract_block import AbstractBlock
from solnml.utils.decorators import time_limit


class AlternatingBlock(AbstractBlock):
    def __init__(self, node_list, node_index,
                 task_type, timestamp,
                 fe_config_space: ConfigurationSpace,
                 cash_config_space: ConfigurationSpace,
                 data: DataNode,
                 fixed_config=None,
                 time_limit=None,
                 trial_num=0,
                 metric='acc',
                 ensemble_method='ensemble_selection',
                 ensemble_size=50,
                 per_run_time_limit=300,
                 output_dir="logs",
                 dataset_name='default_dataset',
                 eval_type='holdout',
                 resampling_params=None,
                 n_jobs=1,
                 seed=1):
        super(AlternatingBlock, self).__init__(node_list, node_index, task_type, timestamp,
                                               fe_config_space, cash_config_space, data,
                                               fixed_config=fixed_config,
                                               time_limit=time_limit,
                                               trial_num=trial_num,
                                               metric=metric,
                                               ensemble_method=ensemble_method,
                                               ensemble_size=ensemble_size,
                                               per_run_time_limit=per_run_time_limit,
                                               output_dir=output_dir,
                                               dataset_name=dataset_name,
                                               eval_type=eval_type,
                                               resampling_params=resampling_params,
                                               n_jobs=n_jobs,
                                               seed=seed)

        self.arms = ['hpo', 'fe']
        self.optimal_algo_id = None
        self.first_start = True
        self.sub_bandits = dict()
        self.rewards = dict()
        self.evaluation_cost = dict()
        self.update_flag = dict()

        # Global incumbent.
        self.init_config = {'fe': fe_config_space.get_default_configuration().get_dictionary().copy(),
                            'hpo': cash_config_space.get_default_configuration().get_dictionary().copy()}
        self.inc = {'fe': fe_config_space.get_default_configuration().get_dictionary().copy(),
                    'hpo': cash_config_space.get_default_configuration().get_dictionary().copy()}
        self.local_inc = {'fe': fe_config_space.get_default_configuration().get_dictionary().copy(),
                          'hpo': cash_config_space.get_default_configuration().get_dictionary().copy()}
        self.local_hist = {'fe': [], 'hpo': []}
        self.inc_record = {'fe': list(), 'hpo': list()}
        self.exp_output = dict()
        self.eval_dict = dict()
        self.arm_eval_dict = {'fe': dict(), 'hpo': dict()}
        for arm in self.arms:
            self.rewards[arm] = list()
            self.update_flag[arm] = False
            self.evaluation_cost[arm] = list()
            self.exp_output[arm] = dict()
        self.pull_cnt = 0
        self.action_sequence = list()
        self.final_rewards = list()

        for arm in self.arms:
            if arm == 'fe':
                from solnml.blocks.block_utils import get_node_type
                child_type = get_node_type(node_list, node_index + 1)
                self.sub_bandits[arm] = child_type(
                    node_list, node_index + 1, task_type, timestamp, fe_config_space, None, data.copy_(),
                    fixed_config=self.init_config['hpo'],
                    time_limit=time_limit,
                    metric=metric,
                    ensemble_method=ensemble_method,
                    ensemble_size=ensemble_size,
                    per_run_time_limit=per_run_time_limit,
                    output_dir=output_dir,
                    dataset_name=dataset_name,
                    eval_type=eval_type,
                    resampling_params=resampling_params,
                    n_jobs=n_jobs,
                    seed=seed
                )
            else:
                from solnml.blocks.block_utils import get_node_type
                child_type = get_node_type(node_list, node_index + 2)
                self.sub_bandits[arm] = child_type(
                    node_list, node_index + 2, task_type, timestamp, None, cash_config_space, data.copy_(),
                    fixed_config=self.init_config['fe'],
                    time_limit=time_limit,
                    metric=metric,
                    ensemble_method=ensemble_method,
                    ensemble_size=ensemble_size,
                    per_run_time_limit=per_run_time_limit,
                    output_dir=output_dir,
                    dataset_name=dataset_name,
                    eval_type=eval_type,
                    resampling_params=resampling_params,
                    n_jobs=n_jobs,
                    seed=seed
                )

        self.topk_saver = CombinedTopKModelSaver(k=50, model_dir=self.output_dir, identifier=self.timestamp)

    def iterate(self, trial_num=10):
        # First choose one arm.
        arm_to_pull = self.arms[self.pull_cnt % 2]
        self.logger.debug('Pulling arm: %s in node %s at %d-th round' % (arm_to_pull, self.node_index, self.pull_cnt))
        if self.first_start is True and arm_to_pull == 'hpo':
            # trial_budget = 20
            trial_budget = 10
            self.first_start = False
        else:
            trial_budget = trial_num

        if self.sub_bandits[arm_to_pull].early_stop_flag:
            arm_to_pull = self.arms[(self.pull_cnt + 1) % 2]
        start_time = time.time()
        reward = self.sub_bandits[arm_to_pull].iterate(trial_num=trial_budget)
        iter_cost = time.time() - start_time
        self.action_sequence.append(arm_to_pull)
        self.pull_cnt += 1

        # Update results after each iteration
        pre_inc_perf = self.incumbent_perf
        for arm_id in self.arms:
            self.update_flag[arm_id] = False
        self.arm_eval_dict[arm_to_pull].update(self.sub_bandits[arm_to_pull].eval_dict)
        self.eval_dict.update(self.sub_bandits[arm_to_pull].eval_dict)
        self.rewards[arm_to_pull].append(reward)
        self.evaluation_cost[arm_to_pull].append(iter_cost)
        self.local_inc[arm_to_pull] = self.sub_bandits[arm_to_pull].incumbent

        # Update global incumbent from FE and HPO.
        if np.isfinite(reward) and reward > self.incumbent_perf:
            cur_inc = self.sub_bandits[arm_to_pull].incumbent
            self.inc[arm_to_pull] = cur_inc
            self.local_hist[arm_to_pull].append(cur_inc)
            self.optimal_algo_id = arm_to_pull
            self.incumbent_perf = reward

            # Alter-HPO strategy: HPO changes if FE changes, FE keeps though HPO changes
            if arm_to_pull == 'fe':
                self.inc['hpo'] = self.init_config['hpo']
            _incumbent = dict()
            _incumbent.update(self.inc['fe'])
            _incumbent.update(self.inc['hpo'])
            self.incumbent = _incumbent.copy()

            arm_id = 'fe' if arm_to_pull == 'hpo' else 'hpo'
            if arm_to_pull == 'fe':
                self.reinitialize(arm_id)
            else:
                # Only reinitialize fe blocks once.
                if len(self.rewards[arm_to_pull]) == 1:
                    self.reinitialize(arm_id)
                    if cur_inc != self.init_config['hpo']:
                        self.logger.info('Initial hp_config for FE has changed!')
                    self.init_config['hpo'] = cur_inc

            # Evaluate joint result here
            # Alter-HPO specific
            if arm_to_pull == 'fe' and self.sub_bandits['fe'].fixed_config != self.local_inc['hpo']:
                self.logger.info("Evaluate joint performance in node %s" % self.node_index)
                self.evaluate_joint_perf()

        # Logger output
        scores = list()
        for _arm in self.arms:
            scores.append(self.sub_bandits[_arm].incumbent_perf)
        scores = np.array(scores)
        self.logger.info('=' * 50)
        self.logger.info('Node index: %s' % str(self.node_index))
        self.logger.info('Best_part_perf: %s' % str(self.incumbent_perf))
        self.logger.info('Best_part: %s' % str(self.optimal_algo_id))
        self.logger.info('Best val scores: %s' % str(list(scores)))
        self.logger.info('=' * 50)

        self.final_rewards.append(self.incumbent_perf)
        post_inc_perf = self.incumbent_perf
        if np.isfinite(pre_inc_perf) and np.isfinite(post_inc_perf):
            self.inc_record[arm_to_pull].append(post_inc_perf - pre_inc_perf)
        else:
            self.inc_record[arm_to_pull].append(0.)

        # Update stop flag
        self.early_stop_flag = True
        self.timeout_flag = False
        for _arm in self.arms:
            if not self.sub_bandits[_arm].early_stop_flag:
                self.early_stop_flag = False
        if self.early_stop_flag:
            self.logger.info(
                "Maximum configuration number met for each arm candidate in alternating block %s!" % self.node_index)
        for _arm in self.arms:
            if self.sub_bandits[_arm].timeout_flag:
                self.timeout_flag = True

        return self.incumbent_perf

    def reinitialize(self, arm_id):
        if arm_id == 'fe':
            # Build the Feature Engineering component.
            inc_hpo = self.inc['hpo'].copy()

            from solnml.blocks.block_utils import get_node_type
            child_type = get_node_type(self.node_list, self.node_index + 1)
            self.sub_bandits[arm_id] = child_type(
                self.node_list, self.node_index + 1, self.task_type,
                self.timestamp, self.fe_config_space, None, self.original_data.copy_(),
                fixed_config=inc_hpo,
                time_limit=self.time_limit,
                metric=self.metric,
                ensemble_method=self.ensemble_method,
                ensemble_size=self.ensemble_size,
                per_run_time_limit=self.per_run_time_limit,
                output_dir=self.output_dir,
                dataset_name=self.dataset_name,
                eval_type=self.eval_type,
                resampling_params=self.resampling_params,
                n_jobs=self.n_jobs,
                seed=self.seed
            )
        else:
            # trials_per_iter = self.optimizer['fe'].evaluation_num_last_iteration // 2
            # trials_per_iter = max(20, trials_per_iter)
            inc_fe = self.inc['fe'].copy()
            from solnml.blocks.block_utils import get_node_type
            child_type = get_node_type(self.node_list, self.node_index + 2)
            self.sub_bandits[arm_id] = child_type(
                self.node_list, self.node_index + 2, self.task_type,
                self.timestamp, None, self.cash_config_space, self.original_data.copy_(),
                fixed_config=inc_fe,
                time_limit=self.time_limit,
                metric=self.metric,
                ensemble_method=self.ensemble_method,
                ensemble_size=self.ensemble_size,
                per_run_time_limit=self.per_run_time_limit,
                output_dir=self.output_dir,
                dataset_name=self.dataset_name,
                eval_type=self.eval_type,
                resampling_params=self.resampling_params,
                n_jobs=self.n_jobs,
                seed=self.seed
            )

        self.logger.debug('=' * 30)
        self.logger.debug('UPDATE OPTIMIZER: %s' % arm_id)
        self.logger.debug('=' * 30)

    # TODO: Need refactoring
    def evaluate_joint_perf(self):
        # Update join incumbent from FE and HPO.
        _perf = None
        try:
            with time_limit(self.per_run_time_limit):
                if self.task_type in CLS_TASKS:
                    from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator
                    evaluator = ClassificationEvaluator(
                        self.local_inc['fe'].copy(),
                        scorer=self.metric,
                        data_node=self.original_data,
                        if_imbal=self.if_imbal,
                        timestamp=self.timestamp,
                        seed=self.seed,
                        output_dir=self.output_dir,
                        resampling_strategy=self.eval_type,
                        resampling_params=self.resampling_params)
                else:
                    from solnml.components.evaluators.rgs_evaluator import RegressionEvaluator
                    evaluator = RegressionEvaluator(
                        self.local_inc['fe'].copy(),
                        scorer=self.metric,
                        data_node=self.original_data,
                        timestamp=self.timestamp,
                        seed=self.seed,
                        output_dir=self.output_dir,
                        resampling_strategy=self.eval_type,
                        resampling_params=self.resampling_params)
                _perf = -evaluator(self.local_inc['hpo'].copy())
        except Exception as e:
            self.logger.error(str(e))

        if _perf is not None and np.isfinite(_perf):
            _config = self.local_inc['fe'].copy()
            _config.update(self.local_inc['hpo'].copy())

            classifier_id = _config['algorithm']
            # -perf: The larger, the better.
            save_flag, model_path, delete_flag, model_path_deleted = self.topk_saver.add(_config, -_perf,
                                                                                         classifier_id)
            # By default, the evaluator has already stored the models.
            if self.eval_type in ['holdout', 'partial']:
                if save_flag:
                    pass
                else:
                    os.remove(model_path)
                    self.logger.info("Model deleted from %s" % model_path)

                try:
                    if delete_flag:
                        os.remove(model_path_deleted)
                        self.logger.info("Model deleted from %s" % model_path_deleted)
                    else:
                        pass
                except:
                    pass
            self.topk_saver.save_topk_config()

        # Update INC.
        if _perf is not None and np.isfinite(_perf) and _perf > self.incumbent_perf:
            self.inc['hpo'] = self.local_inc['hpo']
            self.inc['fe'] = self.local_inc['fe']
            self.incumbent_perf = _perf
            _incumbent = dict()
            _incumbent.update(self.inc['fe'])
            _incumbent.update(self.inc['hpo'])
            self.incumbent = _incumbent.copy()
            # TODO: Add eval_dict
