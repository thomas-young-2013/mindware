import time
from ConfigSpace import ConfigurationSpace

from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.optimizers import build_hpo_optimizer
from mindware.blocks.abstract_block import AbstractBlock


class JointBlock(AbstractBlock):
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
        super(JointBlock, self).__init__(node_list, node_index, task_type, timestamp,
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

        self.fixed_config = fixed_config

        # Combine configuration space
        cs = ConfigurationSpace()
        if fe_config_space is not None:
            cs.add_hyperparameters(fe_config_space.get_hyperparameters())
            cs.add_conditions(fe_config_space.get_conditions())
            cs.add_forbidden_clauses(fe_config_space.get_forbiddens())
        if cash_config_space is not None:
            cs.add_hyperparameters(cash_config_space.get_hyperparameters())
            cs.add_conditions(cash_config_space.get_conditions())
            cs.add_forbidden_clauses(cash_config_space.get_forbiddens())
        self.joint_cs = cs

        # Define evaluator and optimizer
        if self.task_type in CLS_TASKS:
            from mindware.components.evaluators.cls_evaluator import ClassificationEvaluator
            self.evaluator = ClassificationEvaluator(
                fixed_config=fixed_config,
                scorer=self.metric,
                data_node=self.original_data,
                if_imbal=self.if_imbal,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                resampling_strategy=self.eval_type,
                resampling_params=self.resampling_params)
        else:
            from mindware.components.evaluators.rgs_evaluator import RegressionEvaluator
            self.evaluator = RegressionEvaluator(
                fixed_config=fixed_config,
                scorer=self.metric,
                data_node=self.original_data,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                resampling_strategy=self.eval_type,
                resampling_params=self.resampling_params)

        self.optimizer = build_hpo_optimizer(self.eval_type, self.evaluator, self.joint_cs,
                                             output_dir=self.output_dir,
                                             per_run_time_limit=self.per_run_time_limit,
                                             inner_iter_num_per_iter=1,
                                             timestamp=self.timestamp,
                                             seed=self.seed, n_jobs=self.n_jobs)

    def iterate(self, trial_num=10):
        self.optimizer.inner_iter_num_per_iter = trial_num
        self.optimizer.iterate(budget=self.time_limit + self.timestamp - time.time())
        if time.time() - self.timestamp > self.time_limit:
            self.timeout_flag = True
            self.logger.info('Time elapsed!')
        self.early_stop_flag = self.optimizer.early_stopped_flag
        self.incumbent_perf = self.optimizer.incumbent_perf
        self.incumbent = self.optimizer.incumbent_config.get_dictionary().copy()
        self.eval_dict = self.optimizer.eval_dict
        return self.incumbent_perf
