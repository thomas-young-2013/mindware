from sklearn.utils.multiclass import type_of_target
import numpy as np

from automlToolkit.base_estimator import BaseEstimator
from automlToolkit.components.metrics.metric import get_metric
from automlToolkit.components.utils.constants import type_dict, MULTILABEL_CLS
from automlToolkit.components.feature_engineering.transformation_graph import DataNode


class Classifier(BaseEstimator):
    """This class implements the classification task. """

    def fit(self, data: DataNode):
        """
        Fit the classifier to given training data.
        :param data: instance of DataNode
        :return: self
        """
        self.metric = 'acc' if self.metric is None else self.metric

        # Check the task type: {binary, multiclass}
        task_type = type_of_target(data.data[1])
        if task_type in type_dict:
            task_type = type_dict[task_type]
        else:
            raise ValueError("Invalid Task Type: %s!" % task_type)
        self.task_type = task_type
        self.metric = get_metric(self.metric)
        super().fit(data)

        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        """
        Predict classes for X.
        :param X: Datanode
        :param batch_size: int
        :param n_jobs: int
        :return: y : array of shape = [n_samples]
            The predicted classes.
        """
        if not isinstance(X, DataNode):
            raise ValueError("X is supposed to be a Data Node, but get %s" % type(X))
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        """
        Predict probabilities of classes for all samples X.
        :param X: Datanode
        :param batch_size: int
        :param n_jobs: int
        :return: y : array of shape = [n_samples, n_classes]
            The predicted class probabilities.
        """
        if not isinstance(X, DataNode):
            raise ValueError("X is supposed to be a Data Node, but get %s" % type(X))
        pred_proba = super().predict_proba(X, batch_size=batch_size, n_jobs=n_jobs)

        if self.task_type != MULTILABEL_CLS:
            assert (
                np.allclose(
                    np.sum(pred_proba, axis=1),
                    np.ones_like(pred_proba[:, 0]))
            ), "Prediction probability does not sum up to 1!"

        # Check that all probability values lie between 0 and 1.
        assert (
                (pred_proba >= 0).all() and (pred_proba <= 1).all()
        ), "Found prediction probability value outside of [0, 1]!"

        return pred_proba


class Regressor(BaseEstimator):
    """This class implements the regression task. """

    def fit(self, data, **kwargs):
        """
        Fit the regressor to given training data.
        :param data: DataNode
        :return: self
        """
        self.metric = 'mse' if self.metric is None else self.metric

        # Check the task type: {continuous}
        task_type = type_of_target(data.data[1])
        if task_type in type_dict:
            task_type = type_dict[task_type]
        else:
            raise ValueError("Invalid Task Type: %s!" % task_type)
        self.task_type = task_type
        self.metric = get_metric(self.metric)
        super().fit(data)

        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        """
        Make predictions for X.
        :param X: DataNode
        :param batch_size: int
        :param n_jobs: int
        :return: y : array of shape = [n_samples] or [n_samples, n_labels]
            The predicted classes.
        """
        if not isinstance(X, DataNode):
            raise ValueError("X is supposed to be a Data Node, but get %s" % type(X))
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)
