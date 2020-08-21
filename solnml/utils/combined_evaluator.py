from ConfigSpace import ConfigurationSpace, UnParametrizedHyperparameter, CategoricalHyperparameter
import os
import sys
import time
import numpy as np
from sklearn.metrics.scorer import balanced_accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.getcwd())

from solnml.utils.logging_utils import get_logger
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.evaluators.evaluate_func import holdout_validation, cross_validation, partial_validation
from solnml.components.fe_optimizers.ano_bo_optimizer import AnotherBayesianOptimizationOptimizer
from solnml.components.fe_optimizers.bo_optimizer import BayesianOptimizationOptimizer
from solnml.components.evaluators.cls_evaluator import ClassificationEvaluator
from solnml.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from solnml.datasets.utils import load_train_test_data, load_data
from solnml.components.ensemble.combined_ensemble.ensemble_bulider import EnsembleBuilder
from solnml.components.models.classification import _classifiers, _addons


def get_estimator(config):
    classifier_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    config_['random_state'] = 1
    hpo_config = dict()
    for key in config_:
        if 'placeholder' in key:
            act_key = key.split(':')[1]
            hpo_config[act_key] = config_[key]
    try:
        estimator = _classifiers[classifier_type](**hpo_config)
    except:
        estimator = _addons.components[classifier_type](**hpo_config)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', 4)
    return classifier_type, estimator


def get_hpo_cs(estimator_id):
    if estimator_id in _classifiers:
        clf_class = _classifiers[estimator_id]
    elif estimator_id in _addons.components:
        clf_class = _addons.components[estimator_id]
    else:
        raise ValueError("Algorithm %s not supported!" % estimator_id)
    cs = clf_class.get_hyperparameter_search_space()
    return cs


def get_fe_cs(estimator_id):
    tmp_node = load_data('balloon', task_type=0, datanode_returned=True)
    tmp_evaluator = ClassificationEvaluator(None)
    tmp_bo = AnotherBayesianOptimizationOptimizer(0, tmp_node, tmp_evaluator, estimator_id, 1, 1, 1)
    cs = tmp_bo._get_task_hyperparameter_space('smac')
    return cs


def get_combined_cs(estimator_id):
    cs = ConfigurationSpace()
    hpo_cs = get_hpo_cs(estimator_id)
    fe_cs = get_fe_cs(estimator_id)
    config_cand = ['placeholder']
    config_option = CategoricalHyperparameter('hpo', config_cand)
    cs.add_hyperparameter(config_option)
    for config_item in config_cand:
        sub_configuration_space = hpo_cs
        parent_hyperparameter = {'parent': config_option,
                                 'value': config_item}
        cs.add_configuration_space(config_item, sub_configuration_space,
                                   parent_hyperparameter=parent_hyperparameter)
    for hp in fe_cs.get_hyperparameters():
        cs.add_hyperparameter(hp)
    for cond in fe_cs.get_conditions():
        cs.add_condition(cond)
    for bid in fe_cs.get_forbiddens():
        cs.add_forbidden_clause(bid)
    model = UnParametrizedHyperparameter("estimator", estimator_id)
    cs.add_hyperparameter(model)
    return cs


class CombinedEvaluator(_BaseEvaluator):
    def __init__(self, scorer=None, data_node=None,
                 resampling_strategy='cv', resampling_params=None, seed=1):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params
        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.data_node = data_node
        self.seed = seed
        self.eval_id = 0
        self.onehot_encoder = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        tmp_node = load_data('balloon', task_type=0, datanode_returned=True)
        tmp_evaluator = ClassificationEvaluator(None)
        self.tmp_bo = AnotherBayesianOptimizationOptimizer(0, tmp_node, tmp_evaluator, 'adaboost', 1, 1, 1)

    def get_fit_params(self, y, estimator):
        from solnml.components.utils.balancing import get_weights
        _init_params, _fit_params = get_weights(
            y, estimator, None, {}, {})
        return _init_params, _fit_params

    def __call__(self, config, **kwargs):
        start_time = time.time()
        return_dict = dict()

        downsample_ratio = kwargs.get('resource_ratio', 1.0)
        # Prepare data node.
        data_node = self.tmp_bo._parse(self.data_node, config)

        X_train, y_train = data_node.data

        config_dict = config.get_dictionary().copy()
        # Prepare training and initial params for classifier.
        init_params, fit_params = {}, {}
        if data_node.enable_balance == 1:
            init_params, fit_params = self.get_fit_params(y_train, config['estimator'])
            for key, val in init_params.items():
                config_dict[key] = val

        if data_node.data_balance == 1:
            fit_params['data_balance'] = True

        classifier_id, clf = get_estimator(config_dict)

        if self.onehot_encoder is None:
            self.onehot_encoder = OneHotEncoder(categories='auto')
            y = np.reshape(y_train, (len(y_train), 1))
            self.onehot_encoder.fit(y)

        try:
            if 'cv' in self.resampling_strategy:
                if self.resampling_params is None or 'folds' not in self.resampling_params:
                    folds = 5
                else:
                    folds = self.resampling_params['folds']
                score = cross_validation(clf, self.scorer, X_train, y_train,
                                         n_fold=folds,
                                         random_state=1,
                                         if_stratify=True,
                                         onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                  _ThresholdScorer) else None,
                                         fit_params=fit_params)
            elif 'holdout' in self.resampling_strategy:
                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']
                score = holdout_validation(clf, self.scorer, X_train, y_train,
                                           test_size=test_size,
                                           random_state=1,
                                           if_stratify=True,
                                           onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                    _ThresholdScorer) else None,
                                           fit_params=fit_params)
            elif 'partial' in self.resampling_strategy:
                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']
                score = partial_validation(clf, self.scorer, X_train, y_train, downsample_ratio,
                                           test_size=test_size,
                                           random_state=self.seed,
                                           if_stratify=True,
                                           onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                    _ThresholdScorer) else None,
                                           fit_params=fit_params)
            else:
                raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)
        except Exception as e:
            self.logger.info('evaluator: %s' % (str(e)))
            score = -np.inf
        # print(config)
        self.logger.info('%d-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                         (self.eval_id, classifier_id,
                          self.scorer._sign * score,
                          time.time() - start_time, X_train.shape))
        self.eval_id += 1

        # Turn it into a minimization problem.
        return_dict['score'] = -score
        return -score


def fetch_ensemble_members(nbest_algo_ids, seed, eval_dict, record_op):
    stats = dict()
    stats['include_algorithms'] = nbest_algo_ids
    stats['split_seed'] = seed

    for algo_id in nbest_algo_ids:
        data = dict()
        model_num = 60

        _eval_list = sorted(eval_dict[algo_id].items(), key=lambda item: item[1], reverse=True)
        _eval_list = list(filter(lambda item: np.isfinite(item[1]), _eval_list))
        model_items = _eval_list[:model_num]

        configs = [item[0][1] for item in model_items]

        node_list = record_op.fetch_nodes_by_config(configs)
        model_to_eval = []
        for idx, node in enumerate(node_list):
            if node is not None:
                model_to_eval.append((node_list[idx], configs[idx]))
        data['model_to_eval'] = model_to_eval
        stats[algo_id] = data
    return stats


if __name__ == '__main__':
    train_node, test_node = load_train_test_data('abalone', task_type=0)
    seed = 1
    evaluator = CombinedEvaluator(data_node=train_node, seed=seed, scorer=balanced_accuracy_scorer)
    estimator_ids = ['libsvm_svc', 'random_forest']
    op = dict()

    # Get combined space and initialize optimizer
    for estimator_id in estimator_ids:
        cs = get_combined_cs(estimator_id)
        op[estimator_id] = SMACOptimizer(evaluator, cs, inner_iter_num_per_iter=10)

    # Iterate (modify search strategy here)
    for estimator_id in estimator_ids:
        op[estimator_id].iterate()

    # Fetch ensemble members
    eval_dict = dict()
    for estimator_id in estimator_ids:
        eval_dict[estimator_id] = op[estimator_id].eval_dict
    # Important: Specify n_best ids according to search strategy
    nbest_ids = estimator_ids
    tmp_evaluator = ClassificationEvaluator(None)
    tmp_bo = BayesianOptimizationOptimizer(0, train_node, tmp_evaluator, 'adaboost', 1, 1,
                                           1)  # A tmp optimizer for recording fe transformations
    stats = fetch_ensemble_members(nbest_ids, seed, eval_dict, record_op=tmp_bo)

    es = EnsembleBuilder(stats, 'ensemble_selection', 50, task_type=0, metric=balanced_accuracy_scorer,
                         output_dir='logs/')
    es.fit(train_node)
    pred = es.predict(test_node, record_op=tmp_bo)  # Proba output
    cls_pred = np.argmax(pred, axis=-1)

    print(cls_pred)
