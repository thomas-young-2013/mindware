import time
import numpy as np
from sklearn.metrics.scorer import accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

from solnml.utils.logging_utils import get_logger
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.evaluators.img_evaluate_func import img_holdout_validation


def get_estimator(config):
    from solnml.components.models.img_classification import _classifiers, _addons
    classifier_type = config['estimator']
    config_ = config.copy()
    config_.pop('estimator', None)
    config_['random_state'] = 1
    try:
        estimator = _classifiers[classifier_type](**config_)
    except:
        estimator = _addons.components[classifier_type](**config_)
    return classifier_type, estimator


class ImgClassificationEvaluator(_BaseEvaluator):
    def __init__(self, clf_config, scorer=None, dataset=None, seed=1):
        self.hpo_config = clf_config
        self.scorer = scorer if scorer is not None else accuracy_scorer
        self.dataset = dataset
        self.seed = seed
        self.eval_id = 0
        self.onehot_encoder = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

    def __call__(self, config, **kwargs):
        start_time = time.time()

        # Prepare configuration.
        np.random.seed(self.seed)
        config = config if config is not None else self.hpo_config

        # downsample_ratio = kwargs.get('data_subsample_ratio', 1.0)

        config_dict = config.get_dictionary().copy()

        classifier_id, clf = get_estimator(config_dict)

        # if self.onehot_encoder is None:
        #     self.onehot_encoder = OneHotEncoder(categories='auto')
        #     y = np.reshape(y_train, (len(y_train), 1))
        #     self.onehot_encoder.fit(y)

        try:
            score = img_holdout_validation(clf, self.scorer, self.dataset, random_state=self.seed)
            # score = partial_validation(clf, self.scorer, self.dataset, downsample_ratio,
            #                            random_state=self.seed,
            #                            if_stratify=True,
            #                            onehot=self.onehot_encoder if isinstance(self.scorer,
            #                                                                     _ThresholdScorer) else None)
        except Exception as e:
            self.logger.error(e)
            score = np.inf

        self.logger.debug('%d-Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds' %
                          (self.eval_id, classifier_id,
                           self.scorer._sign * score,
                           time.time() - start_time))
        self.eval_id += 1

        # Turn it into a minimization problem.
        return -score
