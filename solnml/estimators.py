import numpy as np
from sklearn.utils.multiclass import type_of_target
from solnml.base_estimator import BaseEstimator, BaseDLEstimator
from solnml.components.utils.constants import type_dict, MULTILABEL_CLS, IMG_CLS
from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.datasets.image_dataset import ImageDataset


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

    def refit(self):
        return super().refit()

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

    def get_tree_importance(self, data: DataNode):
        from lightgbm import LGBMClassifier
        import pandas as pd
        X, y = self.data_transformer(data).data
        lgb = LGBMClassifier(random_state=1)
        lgb.fit(X, y)
        _importance = lgb.feature_importances_
        h = {}
        h['feature_id'] = np.array(range(len(_importance)))
        h['feature_importance'] = _importance
        return pd.DataFrame(h)

    def get_linear_importance(self, data: DataNode):
        from sklearn.linear_model import LogisticRegression
        import pandas as pd
        X, y = self.data_transformer(data).data
        clf = LogisticRegression(random_state=1)
        clf.fit(X, y)
        _ef = clf.coef_
        std_array = np.std(_ef, ddof=1, axis=0)
        abs_array = abs(_ef)
        mean_array = np.mean(abs_array, axis=0)
        _importance = std_array / mean_array
        h = {}
        h['feature_id'] = np.array(range(len(_importance)))
        h['feature_importance'] = _importance
        return pd.DataFrame(h)

    def get_linear_impact(self, data: DataNode):
        from sklearn.linear_model import LogisticRegression
        import pandas as pd
        if (len(set(data.data[1]))) > 2:
            print('ERROR! Only binary classification is supported!')
            return 0
        X, y = self.data_transformer(data).data
        clf = LogisticRegression(random_state=1)
        clf.fit(X, y)
        _ef = clf.coef_
        _impact = _ef[0]
        h = {}
        h['feature_id'] = np.array(range(len(_impact)))
        h['feature_impact'] = _impact
        return pd.DataFrame(h)


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

    def get_tree_importance(self, data: DataNode):
        from lightgbm import LGBMRegressor
        import pandas as pd
        X, y = self.data_transformer(data).data
        lgb = LGBMRegressor(random_state=1)
        lgb.fit(X, y)
        _importance = lgb.feature_importances_
        h = {}
        h['feature_id'] = np.array(range(len(_importance)))
        h['feature_importance'] = _importance
        return pd.DataFrame(h)

    def get_linear_impact(self, data: DataNode):
        from sklearn.linear_model import LinearRegression
        import pandas as pd
        X, y = self.data_transformer(data).data
        reg = LinearRegression()
        reg.fit(X, y)
        _impact = reg.coef_
        h = {}
        h['feature_id'] = np.array(range(len(_impact)))
        h['feature_impact'] = _impact
        return pd.DataFrame(h)


class ImageClassifier(BaseDLEstimator):
    """This class implements the classification task. """

    def fit(self, data: ImageDataset):
        """
        Fit the classifier to given training data.
        :param data: instance of Image Dataset
        :return: self
        """
        self.metric = 'acc' if self.metric is None else self.metric
        # Set task type to image classification.
        self.task_type = IMG_CLS
        super().fit(data)

        return self

    def predict(self, X, batch_size=1, n_jobs=1):
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def refit(self):
        return super().refit()

    def predict_proba(self, X, batch_size=1, n_jobs=1):
        """
        Predict probabilities of classes for all samples X.
        :param X: Datanode
        :param batch_size: int
        :param n_jobs: int
        :return: y : array of shape = [n_samples, n_classes]
            The predicted class probabilities.
        """
        if not isinstance(X, ImageDataset):
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
