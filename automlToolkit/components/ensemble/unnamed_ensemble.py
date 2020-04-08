import numpy as np
from sklearn.metrics.scorer import _BaseScorer
from automlToolkit.components.utils.constants import CLS_TASKS


def choose_base_models(predictions, labels, num_model):
    base_mask = [0] * len(predictions)
    return base_mask


def calculate_weights(predictions, labels, base_mask):
    weights = [0] * len(predictions)
    return weights


class UnnamedEnsemble:
    def __init__(
            self,
            ensemble_size: int,
            task_type: int,
            metric: _BaseScorer,
            random_state: np.random.RandomState = None,
    ):
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.random_state = random_state
        self.base_model_mask = None
        self.weights_ = None

    def fit(self, predictions, labels):
        """

        :param predictions: proba_predictions for cls. Shape: (num_models,num_samples,num_class) for cls
        :param labels: Shape: (num_samples,)
        :return: self
        """
        if self.task_type in CLS_TASKS:  # If classification
            self.base_model_mask = choose_base_models(predictions, labels, self.ensemble_size)
            self.weights_ = calculate_weights(predictions, labels, self.base_model_mask)
        else:
            pass
        return self

    def predict(self, predictions):
        predictions = np.asarray(predictions)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if predictions.shape[0] == len(self.weights_):
            return np.average(predictions, axis=0, weights=self.weights_)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif predictions.shape[0] == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            return np.average(predictions, axis=0, weights=non_null_weights)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
