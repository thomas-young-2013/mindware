import time


class BaseModel(object):
    @staticmethod
    def get_properties():
        """
        Get the properties of the underlying algorithm.
        :return: algorithm_properties : dict, optional (default=None)
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space():
        """
        Get the configuration space of this classification algorithm.
        :return: Configspace.configuration_space.ConfigurationSpace
            The configuration space of this classification algorithm.
        """
        raise NotImplementedError()

    def fit(self, X, y):
        """
        The fit function calls the fit function of the underlying model and returns `self`.
        :param X: array-like, shape = (n_samples, n_features), training data.
        :param y: array-like, shape = (n_samples,) or shape = (n_sample, n_labels).
        :return: self, an instance of self.
        """
        raise NotImplementedError()

    def set_hyperparameters(self, params, init_params=None):
        """
        The function set the class members according to params
        :param params: dictionary, parameters
        :param init_params: dictionary
        :return:
        """
        for param, value in params.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' % (param, str(self)))
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError('Cannot set init param %s for %s because '
                                     'the init param does not exist.' %
                                     (param, str(self)))
                setattr(self, param, value)
        return self


class BaseClassificationModel(BaseModel):
    def __init__(self):
        self.estimator = None
        self.properties = None

    def predict(self, X):
        """
        The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.
        :param X: array-like, shape = (n_samples, n_features).
        :return: the predicted values, array, shape = (n_samples,) or shape = (n_samples, n_labels).
        """
        raise NotImplementedError()

    def predict_proba(self, X):
        """
        Predict probabilities.
        :param X: array-like, shape = (n_samples, n_features).
        :return: array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes).
        """
        raise NotImplementedError()

    def get_estimator(self):
        """
        Return the underlying estimator object.
        :return: the estimator object.
        """
        return self.estimator


class BaseRegressionModel(BaseModel):
    def __init__(self):
        self.estimator = None
        self.properties = None

    def predict(self, X):
        """
        The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.
        :param X: array-like, shape = (n_samples, n_features).
        :return: the predicted values, array, shape = (n_samples,) or shape = (n_samples, n_labels).
        """
        raise NotImplementedError()

    def get_estimator(self):
        """
        Return the underlying estimator object.
        :return: the estimator object.
        """
        return self.estimator


class IterativeComponentWithSampleWeight(BaseModel):
    def fit(self, X, y, sample_weight=None):
        self.iterative_fit(
            X, y, n_iter=2, refit=True, sample_weight=sample_weight
        )
        iteration = 2
        while not self.configuration_fully_fitted():
            n_iter = int(2 ** iteration / 2)
            self.iterative_fit(X, y, n_iter=n_iter, sample_weight=sample_weight)
            iteration += 1
        return self

    @staticmethod
    def get_max_iter():
        raise NotImplementedError()

    def get_current_iter(self):
        raise NotImplementedError()

    # def time_limit_exceeded(self):
    #     if self.time_limit is None:
    #         return False
    #     current_time = time.time()
    #     if current_time - self.start_time > self.time_limit:
    #         return True
    #     else:
    #         return False


class IterativeComponent(BaseModel):
    def fit(self, X, y, sample_weight=None):
        self.iterative_fit(X, y, n_iter=2, refit=True)
        iteration = 2
        while not self.configuration_fully_fitted():
            n_iter = int(2 ** iteration / 2)
            self.iterative_fit(X, y, n_iter=n_iter, refit=False)
            iteration += 1
        return self

    @staticmethod
    def get_max_iter():
        raise NotImplementedError()

    def get_current_iter(self):
        raise NotImplementedError()
