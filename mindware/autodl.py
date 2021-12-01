import os
import time
import torch
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from mindware.components.utils.constants import IMG_CLS, TEXT_CLS, OBJECT_DET
from mindware.datasets.base_dl_dataset import DLDataset
from mindware.components.ensemble.dl_ensemble.ensemble_bulider import EnsembleBuilder
from mindware.components.optimizers import build_hpo_optimizer
from mindware.components.evaluators.dl_evaluator import DLEvaluator
from mindware.components.evaluators.base_dl_evaluator import get_estimator_with_parameters, CombinedTopKModelSaver, \
    get_estimator, get_nas_estimator, get_nas_estimator_with_parameters
from mindware.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_aug_hyperparameter_space, \
    get_test_transforms, get_transforms
from mindware.components.utils.config_parser import ConfigParser
from .autodl_base import AutoDLBase


# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (10240, rlimit[1]))


class ModelSelection(AutoDLBase):
    def __init__(self, time_limit=300,
                 trial_num=None,
                 dataset_name='default_name',
                 task_type=IMG_CLS,
                 metric='acc',
                 include_algorithms=None,
                 ensemble_method='ensemble_selection',
                 ensemble_size=50,
                 max_epoch=150,
                 skip_profile=False,
                 config_file_path=None,
                 evaluation='holdout',
                 logging_config=None,
                 output_dir="logs/",
                 random_state=1,
                 n_jobs=1, **kwargs):
        super().__init__(time_limit=time_limit, trial_num=trial_num, dataset_name=dataset_name, task_type=task_type,
                         metric=metric, include_algorithms=include_algorithms, ensemble_method=ensemble_method,
                         ensemble_size=ensemble_size, max_epoch=max_epoch, config_file_path=config_file_path,
                         evaluation=evaluation, logging_config=logging_config, output_dir=output_dir,
                         random_state=random_state, n_jobs=n_jobs)
        self.skip_profile = skip_profile
        self.timestamp = time.time()

        # SEE algorithm parameters.
        self.optalgo = None
        self.see_optimizer = None

    def fit(self, train_data: DLDataset, **kwargs):
        _start_time = time.time()
        if 'opt_method' in kwargs:
            self.optalgo = kwargs['opt_method']
        else:
            self.optalgo = 'see'

        if self.task_type == IMG_CLS:
            self.image_size = kwargs['image_size']

        if self.config_file_path is not None:
            config_parser = ConfigParser(logger=self.logger)
            self.update_cs = config_parser.read(self.config_file_path)

        # TODO: For first-time user, download pretrained params here!
        algorithm_candidates = self.include_algorithms.copy()
        num_train_samples = train_data.get_train_samples_num()
        if self.optalgo == 'hpo':
            self._fit_in_hpo_way(algorithm_candidates, train_data, **kwargs)
            return

        cs = self.get_pipeline_config_space(self.include_algorithms)
        cs.seed(self.seed)

        self.evaluator = DLEvaluator(self.task_type,
                                     mode='selection',
                                     max_epoch=self.max_epoch,
                                     scorer=self.metric,
                                     dataset=train_data,
                                     device=self.device,
                                     seed=self.seed,
                                     model_dir=self.output_dir,
                                     continue_training=False,
                                     timestamp=self.timestamp,
                                     **kwargs)
        self.optimizer = build_hpo_optimizer(
            'partial_hypertune' if self.evaluation_type == 'partial' else self.evaluation_type,
            self.evaluator, cs,
            output_dir=self.output_dir,
            per_run_time_limit=1000000,
            timestamp=self.timestamp,
            runtime_limit=self.time_limit,
            seed=self.seed, n_jobs=self.n_jobs)

        # Control flow via round robin.
        recorder = self.optimizer.run()

        if len(recorder.incumbents) == 0:
            raise ValueError("No configuration is fully evaluated!")

        self.best_algo_config = recorder.incumbents[0][0].get_dictionary().copy()

        # Skip Ensemble
        if self.task_type == OBJECT_DET:
            return

        if self.ensemble_method is not None:
            stats = self.fetch_ensemble_members(algorithm_candidates)

            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      max_epoch=self.max_epoch,
                                      metric=self.metric,
                                      timestamp=self.timestamp,
                                      device=self.device,
                                      output_dir=self.output_dir, **kwargs)
            self.es.fit(data=train_data)

    def fetch_ensemble_members(self, candidate_algorithms):
        stats = dict()
        # a subset of included algorithms.
        stats['include_algorithms'] = candidate_algorithms
        stats['split_seed'] = self.seed

        self.logger.info('Choose basic models for ensemble stage.')
        self.logger.info('algorithm_id, #models')

        topk_list = CombinedTopKModelSaver.get_topk_config(
            os.path.join(self.output_dir, '%s_topk_config.pkl' % self.timestamp))

        for algo_id in stats['include_algorithms']:
            data = dict()
            leap = 2
            model_num, min_model_num = 20, 5

            elements_list = topk_list[algo_id]

            # topk_configs = [element[0] for element in TopKModelSaver.get_topk_config(
            #     os.path.join('data/dl_models/', '%s_topk_config.pkl' % self.timestamp))]
            #
            # intersection_dict = dict()
            # for key in hpo_eval_dict:
            #     if key[1].get_dictionary() in topk_configs:
            #         intersection_dict[key] = hpo_eval_dict[key]

            hpo_eval_list = filter(lambda item: item[1] != -np.inf, elements_list)
            hpo_eval_list = sorted(hpo_eval_list, key=lambda item: item[1], reverse=True)
            model_items = list()

            if len(hpo_eval_list) > 20:
                idxs = np.arange(min_model_num) * leap
                for idx in idxs:
                    model_items.append(hpo_eval_list[idx])
            else:
                model_items.extend(hpo_eval_list[:min_model_num])

            model_configs = [item[0] for item in model_items]
            data['model_configs'] = model_configs
            self.logger.info('%s, %d' % (algo_id, len(model_configs)))
            stats[algo_id] = data
        self.logger.info('Preparing basic models finished.')
        return stats

    def refit(self, dataset: DLDataset):
        if self.es is None:
            config_dict = self.best_algo_config.get_dictionary().copy()
            model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, self.best_algo_config,
                                                                   self.timestamp)
            # Remove the old models.
            if os.path.exists(model_path):
                os.remove(model_path)

            mode = 'refit'
            if self.task_type == IMG_CLS:
                train_transforms = get_transforms(self.best_algo_config, image_size=self.image_size)
                dataset.load_data(train_transforms['train'], train_transforms['val'])
                if dataset.test_data_path is not None:
                    test_transforms = get_test_transforms(self.best_algo_config, image_size=self.image_size)
                    dataset.load_test_data(test_transforms)
                    mode = 'refit_test'

            else:
                dataset.load_data()
                if dataset.test_data_path is not None:
                    dataset.load_test_data()
                    mode = 'refit_test'

            # Refit the models.
            _, estimator = get_estimator(self.task_type, config_dict, self.max_epoch, device=self.device)
            estimator.fit(dataset, mode=mode)
            # Save to the disk.
            state = {'model': estimator.model.state_dict(),
                     'optimizer': estimator.optimizer_.state_dict(),
                     'scheduler': estimator.scheduler.state_dict(),
                     'epoch_num': estimator.epoch_num,
                     'early_stop': estimator.early_stop}
            torch.save(state, model_path)
        else:
            self.es.refit(dataset)

    def load_predict_data(self, test_data: DLDataset):
        if self.task_type == IMG_CLS:
            test_transforms = get_test_transforms(self.best_algo_config, image_size=self.image_size)
            test_data.load_test_data(test_transforms)
            test_data.load_data(test_transforms, test_transforms)
        else:
            test_data.load_test_data()
            test_data.load_data()

    def predict_proba(self, test_data: DLDataset, mode='test', batch_size=None, n_jobs=1):
        if self.es is None:
            self.load_predict_data(test_data)
            model_ = get_estimator_with_parameters(self.task_type, self.best_algo_config, self.max_epoch,
                                                   test_data.test_dataset, self.timestamp, device=self.device,
                                                   model_dir=self.output_dir)
            if mode == 'test':
                return model_.predict_proba(test_data.test_dataset, batch_size=batch_size)
            else:
                if test_data.subset_sampler_used:
                    return model_.predict_proba(test_data.train_dataset, sampler=test_data.val_sampler,
                                                batch_size=batch_size)
                else:
                    return model_.predict_proba(test_data.val_dataset, batch_size=batch_size)
        else:
            return self.es.predict(test_data, mode=mode)

    def predict(self, test_data: DLDataset, mode='test', batch_size=None, n_jobs=1):
        if self.es is None:
            self.load_predict_data(test_data)
            model_ = get_estimator_with_parameters(self.task_type, self.best_algo_config, self.max_epoch,
                                                   test_data.test_dataset, self.timestamp, device=self.device,
                                                   model_dir=self.output_dir)
            if mode == 'test':
                return model_.predict(test_data.test_dataset, batch_size=batch_size)
            else:
                if test_data.subset_sampler_used:
                    return model_.predict(test_data.train_dataset, sampler=test_data.val_sampler,
                                          batch_size=batch_size)
                else:
                    return model_.predict(test_data.val_dataset, batch_size=batch_size)
        else:
            return np.argmax(self.es.predict(test_data, mode=mode), axis=-1)

    def score(self, test_data: DLDataset, mode='test', metric_func=None):
        if metric_func is None:
            metric_func = self.metric
        preds = self.predict(test_data, mode=mode)
        labels = test_data.get_labels(mode=mode)
        # TODO: support AUC
        return metric_func._score_func(preds, labels)

    def get_pipeline_config_space(self, algorithm_candidates):
        cs = ConfigurationSpace()
        estimator_choice = CategoricalHyperparameter("algorithm", algorithm_candidates,
                                                     default_value=algorithm_candidates[0])
        cs.add_hyperparameter(estimator_choice)
        if self.task_type == IMG_CLS:
            aug_space = get_aug_hyperparameter_space()
            cs.add_hyperparameters(aug_space.get_hyperparameters())
            cs.add_conditions(aug_space.get_conditions())

        for estimator_id in algorithm_candidates:
            sub_cs = self.get_model_config_space(estimator_id, include_estimator=False, include_aug=False)
            parent_hyperparameter = {'parent': estimator_choice,
                                     'value': estimator_id}
            cs.add_configuration_space(estimator_id, sub_cs,
                                       parent_hyperparameter=parent_hyperparameter)
        return cs

    def _fit_in_hpo_way(self, algorithm_candidates, train_data, **kwargs):
        cs = self.get_pipeline_config_space(algorithm_candidates)
        hpo_evaluator = DLEvaluator(self.task_type,
                                    mode='selection',
                                    max_epoch=self.max_epoch,
                                    scorer=self.metric,
                                    dataset=train_data,
                                    device=self.device,
                                    image_size=self.image_size,
                                    continue_training=True if self.evaluation_type == 'partial' else False,
                                    seed=self.seed,
                                    timestamp=self.timestamp)
        optimizer = build_hpo_optimizer(self.evaluation_type, hpo_evaluator, cs,
                                        output_dir=self.output_dir,
                                        per_run_time_limit=1000000,
                                        timestamp=self.timestamp,
                                        seed=self.seed, n_jobs=self.n_jobs)
        self.solvers['hpo_solver'] = optimizer
        self.evaluators['hpo_solver'] = hpo_evaluator

        # Control flow via round robin.
        _start_time = time.time()
        if self.trial_num is None:
            while True:
                _time_elapsed = time.time() - _start_time
                if _time_elapsed >= self.time_limit:
                    break
                _budget_left = self.time_limit - _time_elapsed
                self.solvers['hpo_solver'].iterate(budget=_budget_left)
        else:
            for _ in self.trial_num:
                self.solvers['hpo_solver'].iterate()

        # Best model id.
        self.best_algo_id = 'hpo_solver'
        # Best model configuration.
        solver_ = self.solvers[self.best_algo_id]
        inc_idx = np.argmax(solver_.perfs)
        self.best_algo_config = solver_.configs[inc_idx]

        # Skip Ensemble
        if self.task_type == OBJECT_DET:
            return

        if self.ensemble_method is not None:
            stats = self.fetch_ensemble_members(algorithm_candidates)

            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      max_epoch=self.max_epoch,
                                      metric=self.metric,
                                      device=self.device,
                                      output_dir=self.output_dir,
                                      mode='selection',
                                      **kwargs)
            self.es.fit(data=train_data)

    def _get_runtime_info(self):
        if self.optalgo == 'see':
            return self.see_optimizer.get_evaluation_stats()
        else:
            return self.solvers['hpo_solver'].get_evaluation_stats()


class ModelSearch(AutoDLBase):
    def __init__(self, time_limit=300,
                 trial_num=None,
                 dataset_name='default_name',
                 task_type=IMG_CLS,
                 metric='acc',
                 include_algorithms=None,
                 space='nasbench101',
                 ensemble_method='ensemble_selection',
                 ensemble_size=50,
                 max_epoch=150,
                 skip_profile=False,
                 config_file_path=None,
                 evaluation='holdout',
                 logging_config=None,
                 output_dir="logs/",
                 random_state=1,
                 n_jobs=1, **kwargs):
        super().__init__(time_limit=time_limit, trial_num=trial_num, dataset_name=dataset_name, task_type=task_type,
                         metric=metric, include_algorithms=include_algorithms, ensemble_method=ensemble_method,
                         ensemble_size=ensemble_size, max_epoch=max_epoch, config_file_path=config_file_path,
                         evaluation=evaluation, logging_config=logging_config, output_dir=output_dir,
                         random_state=random_state, n_jobs=n_jobs)
        self.skip_profile = skip_profile
        self.timestamp = time.time()
        self.space = space

    def fit(self, train_data: DLDataset, **kwargs):
        _start_time = time.time()

        if self.task_type == IMG_CLS:
            self.image_size = kwargs['image_size']

        if self.config_file_path is not None:
            config_parser = ConfigParser(logger=self.logger)
            self.update_cs = config_parser.read(self.config_file_path)

        num_train_samples = train_data.get_train_samples_num()

        cs = self.get_config_space()
        cs.seed(self.seed)

        self.evaluator = DLEvaluator(self.task_type,
                                     mode='search',
                                     max_epoch=self.max_epoch,
                                     scorer=self.metric,
                                     dataset=train_data,
                                     device=self.device,
                                     seed=self.seed,
                                     model_dir=self.output_dir,
                                     continue_training=False,
                                     timestamp=self.timestamp,
                                     **kwargs)

        self.optimizer = build_hpo_optimizer(
            'partial_hypertune' if self.evaluation_type == 'partial' else self.evaluation_type,
            self.evaluator, cs,
            output_dir=self.output_dir,
            per_run_time_limit=1000000,
            timestamp=self.timestamp,
            runtime_limit=self.time_limit,
            seed=self.seed, n_jobs=self.n_jobs)

        # Control flow via round robin.
        recorder = self.optimizer.run()

        if len(recorder.incumbents) == 0:
            raise ValueError("No configuration is fully evaluated!")

        self.best_algo_config = recorder.incumbents[0][0].get_dictionary().copy()

        # Skip Ensemble
        if self.task_type == OBJECT_DET:
            return

        if self.ensemble_method is not None:
            stats = self.fetch_ensemble_members([self.space])

            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      max_epoch=self.max_epoch,
                                      metric=self.metric,
                                      timestamp=self.timestamp,
                                      device=self.device,
                                      output_dir=self.output_dir,
                                      mode='search',
                                      **kwargs)
            self.es.fit(data=train_data)

    def fetch_ensemble_members(self, candidate_algorithms):
        stats = dict()
        # a subset of included algorithms.
        stats['include_algorithms'] = candidate_algorithms
        stats['split_seed'] = self.seed

        self.logger.info('Choose basic models for ensemble stage.')
        self.logger.info('algorithm_id, #models')

        topk_list = CombinedTopKModelSaver.get_topk_config(
            os.path.join(self.output_dir, '%s_topk_config.pkl' % self.timestamp))

        for algo_id in stats['include_algorithms']:
            data = dict()
            leap = 2
            model_num, min_model_num = 20, 5

            elements_list = topk_list[algo_id]

            # topk_configs = [element[0] for element in TopKModelSaver.get_topk_config(
            #     os.path.join('data/dl_models/', '%s_topk_config.pkl' % self.timestamp))]
            #
            # intersection_dict = dict()
            # for key in hpo_eval_dict:
            #     if key[1].get_dictionary() in topk_configs:
            #         intersection_dict[key] = hpo_eval_dict[key]

            hpo_eval_list = filter(lambda item: item[1] != -np.inf, elements_list)
            hpo_eval_list = sorted(hpo_eval_list, key=lambda item: item[1], reverse=True)
            model_items = list()

            if len(hpo_eval_list) > 20:
                idxs = np.arange(min_model_num) * leap
                for idx in idxs:
                    model_items.append(hpo_eval_list[idx])
            else:
                model_items.extend(hpo_eval_list[:min_model_num])

            model_configs = [item[0] for item in model_items]
            data['model_configs'] = model_configs
            self.logger.info('%s, %d' % (algo_id, len(model_configs)))
            stats[algo_id] = data
        self.logger.info('Preparing basic models finished.')
        return stats

    def refit(self, dataset: DLDataset):
        if self.es is None:
            config_dict = self.best_algo_config.get_dictionary().copy()
            model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, self.best_algo_config,
                                                                   self.timestamp)
            # Remove the old models.
            if os.path.exists(model_path):
                os.remove(model_path)

            mode = 'refit'
            if self.task_type == IMG_CLS:
                train_transforms = get_transforms(self.best_algo_config, image_size=self.image_size)
                dataset.load_data(train_transforms['train'], train_transforms['val'])
                if dataset.test_data_path is not None:
                    test_transforms = get_test_transforms(self.best_algo_config, image_size=self.image_size)
                    dataset.load_test_data(test_transforms)
                    mode = 'refit_test'

            else:
                dataset.load_data()
                if dataset.test_data_path is not None:
                    dataset.load_test_data()
                    mode = 'refit_test'

            # Refit the models.
            _, estimator = get_nas_estimator(config_dict, self.max_epoch, device=self.device)
            estimator.fit(dataset, mode=mode)
            # Save to the disk.
            state = {'model': estimator.model.state_dict(),
                     'optimizer': estimator.optimizer_.state_dict(),
                     'scheduler': estimator.scheduler.state_dict(),
                     'epoch_num': estimator.epoch_num,
                     'early_stop': estimator.early_stop}
            torch.save(state, model_path)
        else:
            self.es.refit(dataset)

    def load_predict_data(self, test_data: DLDataset):
        if self.task_type == IMG_CLS:
            test_transforms = get_test_transforms(self.best_algo_config, image_size=self.image_size)
            test_data.load_test_data(test_transforms)
            test_data.load_data(test_transforms, test_transforms)
        else:
            test_data.load_test_data()
            test_data.load_data()

    def predict_proba(self, test_data: DLDataset, mode='test', batch_size=None, n_jobs=1):
        if self.es is None:
            self.load_predict_data(test_data)
            model_ = get_nas_estimator_with_parameters(self.best_algo_config, self.max_epoch,
                                                       test_data.test_dataset, self.timestamp, device=self.device,
                                                       model_dir=self.output_dir)
            if mode == 'test':
                return model_.predict_proba(test_data.test_dataset, batch_size=batch_size)
            else:
                if test_data.subset_sampler_used:
                    return model_.predict_proba(test_data.train_dataset, sampler=test_data.val_sampler,
                                                batch_size=batch_size)
                else:
                    return model_.predict_proba(test_data.val_dataset, batch_size=batch_size)
        else:
            return self.es.predict(test_data, mode=mode)

    def predict(self, test_data: DLDataset, mode='test', batch_size=None, n_jobs=1):
        if self.es is None:
            self.load_predict_data(test_data)
            model_ = get_nas_estimator_with_parameters(self.best_algo_config, self.max_epoch,
                                                       test_data.test_dataset, self.timestamp, device=self.device,
                                                       model_dir=self.output_dir)
            if mode == 'test':
                return model_.predict(test_data.test_dataset, batch_size=batch_size)
            else:
                if test_data.subset_sampler_used:
                    return model_.predict(test_data.train_dataset, sampler=test_data.val_sampler,
                                          batch_size=batch_size)
                else:
                    return model_.predict(test_data.val_dataset, batch_size=batch_size)
        else:
            return np.argmax(self.es.predict(test_data, mode=mode), axis=-1)

    def score(self, test_data: DLDataset, mode='test', metric_func=None):
        if metric_func is None:
            metric_func = self.metric
        preds = self.predict(test_data, mode=mode)
        labels = test_data.get_labels(mode=mode)
        # TODO: support AUC
        return metric_func._score_func(preds, labels)

    def get_config_space(self):
        assert self.space in ['nasbench101', 'nasbench201', 'darts']
        from mindware.components.models.search import _searchers
        cs = ConfigurationSpace()
        estimator_choice = CategoricalHyperparameter("algorithm", [self.space],
                                                     default_value=self.space)
        cs.add_hyperparameter(estimator_choice)
        arch_space = _searchers[self.space].get_hyperparameter_search_space()

        cs.add_hyperparameters(arch_space.get_hyperparameters())
        cs.add_conditions(arch_space.get_conditions())
        cs.add_forbidden_clauses(arch_space.get_forbiddens())
        return cs
