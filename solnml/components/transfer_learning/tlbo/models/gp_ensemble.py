import typing
import numpy as np

from ..config_space import ConfigurationSpace
from ..config_space.util import convert_configurations_to_array
from ..models.base_model import BaseModel
from .gp import GaussianProcess
from .gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel
from .gp_base_priors import LognormalPrior, HorseshoePrior
from ..utils.util_funcs import get_types, get_rng


def create_gp_model(config_space, rng=None):
    """
        Construct the Gaussian process model that is capable of dealing with categorical hyperparameters.
    """
    if rng is None:
        _, rng = get_rng(rng)
    types, bounds = get_types(config_space, instance_features=None)

    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
    )

    cont_dims = np.nonzero(types == 0)[0]
    cat_dims = np.nonzero(types != 0)[0]

    if len(cont_dims) > 0:
        exp_kernel = Matern(
            np.ones([len(cont_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
            nu=2.5,
            operate_on=cont_dims,
        )

    if len(cat_dims) > 0:
        ham_kernel = HammingKernel(
            np.ones([len(cat_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
            operate_on=cat_dims,
        )

    noise_kernel = WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(np.exp(-25), np.exp(2)),
        prior=HorseshoePrior(scale=0.1, rng=rng),
    )

    if len(cont_dims) > 0 and len(cat_dims) > 0:
        # both
        kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
    elif len(cont_dims) > 0 and len(cat_dims) == 0:
        # only cont
        kernel = cov_amp * exp_kernel + noise_kernel
    elif len(cont_dims) == 0 and len(cat_dims) > 0:
        # only cont
        kernel = cov_amp * ham_kernel + noise_kernel
    else:
        raise ValueError()

    seed = rng.randint(0, 2 ** 20)
    model = GaussianProcess(config_space, types, bounds, seed, kernel)
    return model


class GaussianProcessEnsemble(BaseModel):
    """
    Gaussian process model ensemble.

    Parameters
    ----------
    types : np.ndarray (D)
        Specifies the number of categorical values of an input dimension where
        the i-th entry corresponds to the i-th input dimension. Let's say we
        have 2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass np.array([2, 0]). Note that we count starting from 0.
    bounds : list
        Specifies the bounds for continuous features.
    seed : int
        Model seed.
    kernel : george kernel object
        Specifies the kernel that is used for all Gaussian Process
    prior : prior object
        Defines a prior for the hyperparameters of the GP. Make sure that
        it implements the Prior interface.
    normalize_y : bool
        Zero mean unit variance normalization of the output values
    rng: np.random.RandomState
        Random number generator
    """

    def __init__(
            self,
            past_runhistory: typing.List,
            configspace: ConfigurationSpace,
            seed: int=1,
            **kwargs
    ):
        _, rng = get_rng(seed)
        types, bounds = get_types(configspace, instance_features=None)
        self.past_runhistory = past_runhistory
        super().__init__(configspace=configspace, types=types, bounds=bounds, seed=seed, **kwargs)

        self.n_init_configs = list()
        self.gp_models = list()
        self.target_model = None
        self.model_weights = None
        self.ignore_flag = None
        self._init()

    def _init(self):
        for _runhistory in self.past_runhistory:
            gp_model = create_gp_model(self.configspace)
            X = list()
            for row in _runhistory:
                conf_vector = convert_configurations_to_array([row[0]])[0]
                X.append(conf_vector)
            X = np.array(X)
            y = np.array([row[1] for row in _runhistory]).reshape(-1, 1)

            gp_model.train(X, y)
            self.gp_models.append(gp_model)
            print('Training basic GP model finished.')

        self.ignore_flag = [False] * len(self.past_runhistory)

    def _update_weights(self):
        self.model_weights = 1./np.ones(len(self.past_runhistory) + 1)

    def _train(self, X: np.ndarray, y: np.ndarray):
        """

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        """

        X = self._impute_inactive(X)
        self.target_model = create_gp_model(self.configspace)
        self.target_model.train(X, y)

        self.is_trained = True
        self._update_weights()

    def _predict(self, X_test: np.ndarray):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.
          Predict the given x's objective value (mean, std).
          The predicting result is influenced by the ensemble surrogate with weights.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,) or np.array(N, N) if full_cov == True
            predictive variance

        """

        n = X_test.shape[0]
        mu, var = np.zeros(n), np.zeros(n)

        if self.target_model is not None:
            mu, var = self.target_model.predict(X_test)

        # Target surrogate predictions with weight.
        mu *= self.model_weights[-1]
        var *= np.power(self.model_weights[-1], 2)

        # Base surrogate predictions with corresponding weights.
        for i in range(0, len(self.past_runhistory)):
            if not self.ignore_flag[i]:
                _w = self.model_weights[i]
                _mu, _var = self.gp_models[i].predict(X_test)
                mu += _w * _mu
                var += _w * _w * _var
        return mu, var
