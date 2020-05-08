import os
import numpy as np
import pickle as pk
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import StratifiedKFold

from litebo.config_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from litebo.facade.bo_facade import BayesianOptimization

meta_algo = 'lightgbm'
meta_dir = 'data/meta_res_cp/'
hash_id = '0113dca840d96fa72e427ab7b6f1d888'
meta_dataset_filename = meta_dir + 'ranker_dataset_%s_%s.pkl' % (meta_algo, hash_id)
if os.path.exists(meta_dataset_filename):
    with open(meta_dataset_filename, 'rb') as f:
        meta_X, meta_y, meta_infos = pk.load(f)
print('meta instance num: %d' % len(meta_y))
scorer = make_scorer(accuracy_score)
X, y = np.array(meta_X), np.array(meta_y)


def objective_function(config):
    gbm = lgb.LGBMClassifier(**config)

    scores = list()
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    for train_idx, valid_idx in kfold.split(X, y):
        train_x, valid_x = X[train_idx], X[valid_idx]
        train_y, valid_y = y[train_idx], y[valid_idx]
        gbm.fit(train_x, train_y)
        # print(train_y, valid_y)
        # pred_y = gbm.predict(valid_x)
        # print(pred_y)
        # scores.append(accuracy_score(valid_y, pred_y))
        scores.append(scorer(gbm, valid_x, valid_y))
    print(-np.mean(scores))
    return -np.mean(scores)


def build_configspace():
    cs = ConfigurationSpace()
    n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=250)
    num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 1023, default_value=31)
    learning_rate = UniformFloatHyperparameter("learning_rate", 0.025, 0.3, default_value=0.1, log=True)
    min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
    subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=1)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.5, 1, default_value=1)
    reg_alpha = UniformFloatHyperparameter('reg_alpha', 1e-10, 10, log=True, default_value=1e-10)
    reg_lambda = UniformFloatHyperparameter("reg_lambda", 1e-10, 10, log=True, default_value=1e-10)
    cs.add_hyperparameters([n_estimators, num_leaves, learning_rate, min_child_weight, subsample,
                            colsample_bytree, reg_alpha, reg_lambda])
    return cs


def tune_meta_learner():
    cs = build_configspace()
    def_value = objective_function(cs.get_default_configuration())
    print("Default Value: %.2f" % (def_value))

    bo = BayesianOptimization(objective_function, cs, max_runs=50, time_limit_per_trial=150)
    bo.run()
    inc_value = bo.get_incumbent()
    config = inc_value[0][0]

    with open(meta_dir + 'meta_learner_%s_%s_config.pkl' % (meta_algo, hash_id), 'wb') as f:
        pk.dump(config, f)
    print('Best hyperparameter config found', config)
    return config


if __name__ == "__main__":
    tune_meta_learner()
