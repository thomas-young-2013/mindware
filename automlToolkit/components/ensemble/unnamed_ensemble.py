import numpy as np
import pandas as pd
from sklearn.metrics.scorer import _BaseScorer
from automlToolkit.components.utils.constants import CLS_TASKS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score


def choose_base_models(predictions, num_model, interval=20):
    num_class = predictions.shape[2]
    num_total_models = predictions.shape[0]
    base_mask = [0] * len(predictions)
    bucket = np.arange(interval+1) / interval
    distribution = []
    for prediction in predictions:
        freq_array = []
        for i in range(num_class):
            class_i = prediction[:, i]
            group = pd.cut(class_i, bucket, right=False)
            counts = group.value_counts()
            freq = list(counts / counts.sum())
            freq_array += freq
        distribution.append(freq_array)  # Shape: (num_total_models,20*num_class)
    distribution = np.array(distribution)

    # Apply the clustering algorithm
    Model = AgglomerativeClustering(n_clusters=num_model, linkage="complete")
    cluster = Model.fit(distribution)
    """
    Select models which are the most nearest to the clustering center
    selected_models = []
    """
    for cluster_label in range(num_model):
        cluster_center = np.zeros(distribution.shape[1])
        count = 0
        """
         Averaging the distribution which belong the same clustering class
          and then get the corresponding distribution center
        """
        for i in range(num_total_models):
            if cluster.labels_[i] == cluster_label:
                count += 1
                cluster_center += distribution[i]
        cluster_center = cluster_center / count
        distances = np.sqrt(np.sum(np.asarray(cluster_center - distribution) ** 2, axis=1))
        selected_model = distances.argmin()
        base_mask[selected_model] = 1

    return base_mask


def calculate_weights(predictions, labels, base_mask):
    num_total_models = predictions.shape[0]
    num_samples = predictions.shape[1]
    weights = np.zeros((num_samples, num_total_models))
    for i in range(num_total_models):
        if base_mask[i] != 0:
            predicted_labels = np.argmax(predictions[i], 1)
            acc = accuracy_score(predicted_labels, labels)
            model_weight = 0.5 * np.log(acc / (1 - acc))  # a concrete value
            shannon_ent = -1.0 * np.sum(predictions[i] * np.log2(predictions[i]), 1)  # shape: (1, num_samples)
            confidence = 1 / np.exp(shannon_ent)
            model_weight = model_weight * confidence  # The weight of current model to all samples
            model_weight = model_weight.reshape(num_samples, 1)
            weights[:, i] = model_weight
    return weights

def calculate_weights_simple(predictions, labels, base_mask):
    num_total_models = predictions.shape[0]
    weights = [0]*num_total_models
    for i in range(num_total_models):
        if base_mask[i] != 0:
            predicted_labels = np.argmax(predictions[i], 1)
            acc = accuracy_score(predicted_labels, labels)
            model_weight = 0.5 * np.log(acc / (1 - acc))  # a concrete value
            weights[i] = model_weight
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
