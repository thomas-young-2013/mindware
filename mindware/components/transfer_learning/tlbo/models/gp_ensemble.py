import time
import typing
import numpy as np

from ..config_space import ConfigurationSpace
from ..config_space.util import convert_configurations_to_array
from ..models.base_model import BaseModel
from .gp import GaussianProcess
from .gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel
from .gp_base_priors import LognormalPrior, HorseshoePrior
from ..utils.util_funcs import get_types, get_rng
from ..models.rf_with_instances import RandomForestWithInstances
from ..utils.constants import MAXINT


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
    model = GaussianProcess(config_space, types, bounds, seed, kernel, return_normalized_y=True)
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
            configspace: ConfigurationSpace,
            gp_models: typing.List,
            gp_fusion: str = 'indp-aspt',
            surrogate_model: str = 'prob_rf',
            n_steps_update: int = 2,
            seed: int = 1,
            **kwargs
    ):
        _, rng = get_rng(seed)
        types, bounds = get_types(configspace, instance_features=None)
        self.surrogate_model = surrogate_model
        self.gp_fusion = gp_fusion
        self.n_steps_update = n_steps_update
        assert self.gp_fusion in ['indp-aspt', 'gpoe', 'no-unct']
        super().__init__(configspace=configspace, types=types, bounds=bounds, seed=seed, **kwargs)

        self.n_init_configs = list()
        self.target_model = None
        self.model_weights = None
        self.ignore_flag = None
        self.gp_models = gp_models
        assert self.gp_models is not None
        self.weight_update_id = 0
        self._init()

    def create_basic_model(self):
        if self.surrogate_model == 'gp':
            _model = create_gp_model(self.configspace)
        else:
            _model = RandomForestWithInstances(self.configspace,
                                               normalize_y=True,
                                               seed=self.rng.randint(MAXINT))
        return _model

    def _init(self):
        self.n_runhistory = len(self.gp_models)
        self.ignore_flag = [False] * self.n_runhistory
        # Set initial weights.
        self.model_weights = np.array([1]*self.n_runhistory + [0]) / self.n_runhistory

    def _update_weights(self, X: np.ndarray, y: np.ndarray):
        _start_time = time.time()
        n_instance = X.shape[0]
        n_fold = 5
        predictive_mu, predictive_std = list(), list()

        for _model in self.gp_models:
            _mu, _var = _model.predict(X)
            predictive_mu.append(_mu)
            predictive_std.append(np.sqrt(_var))

        skip_target_model = True if n_instance < n_fold else False

        if not skip_target_model:
            fold_num = n_instance // n_fold
            target_mu, target_std = list(), list()
            for i in range(n_fold):
                instance_indexs = list(range(n_instance))
                bound = (n_instance - i * fold_num) if i == (n_fold - 1) else fold_num
                start_id = i * fold_num
                del instance_indexs[start_id: start_id+bound]
                _target_model = self.create_basic_model()
                _target_model.train(X[instance_indexs, :], y[instance_indexs])
                _mu, _var = _target_model.predict(X[start_id: start_id+bound])
                target_mu.extend(_mu.flatten())
                target_std.extend(np.sqrt(_var).flatten())
                # target_mu.append(_mu)
                # target_std.append(np.sqrt(_var))

            predictive_mu.append(target_mu)
            predictive_std.append(target_std)

        n_sampling = 100
        argmin_cnt = [0] * (self.n_runhistory + 1)
        ranking_loss_hist = list()

        for _ in range(n_sampling):
            ranking_loss_list = list()
            for task_id in range(self.n_runhistory):
                sampled_y = np.random.normal(predictive_mu[task_id], predictive_std[task_id])
                rank_loss = 0
                for i in range(len(y)):
                    for j in range(len(y)):
                        if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                            rank_loss += 1
                ranking_loss_list.append(rank_loss)

            # Compute ranking loss for target surrogate.
            rank_loss = 0
            if not skip_target_model:
                # fold_num = n_instance // n_fold
                # for i in range(n_fold):
                #     sampled_y = np.random.normal(predictive_mu[self.n_runhistory][i], predictive_std[self.n_runhistory][i])
                #     bound = (n_instance - i * fold_num) if i == (n_fold - 1) else fold_num
                #     start_id = fold_num*i
                #     for i in range(start_id, start_id + bound):
                #         for j in range(n_instance):
                #             if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                #                 rank_loss += 1
                sampled_y = np.random.normal(predictive_mu[self.n_runhistory], predictive_std[self.n_runhistory])
                for i in range(len(y)):
                    for j in range(len(y)):
                        if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                            rank_loss += 1
            else:
                rank_loss = len(y) * len(y)
            ranking_loss_list.append(rank_loss)
            ranking_loss_hist.append(ranking_loss_list)
            argmin_id = np.argmin(ranking_loss_list)
            argmin_cnt[argmin_id] += 1

        self.model_weights = np.array(argmin_cnt) / n_sampling
        print(self.model_weights)

        self.ignore_flag = [False] * self.n_runhistory
        ranking_loss_hist = np.array(ranking_loss_hist)
        threshold = sorted(ranking_loss_hist[:, -1])[int(n_sampling * 0.7)]
        for i in range(self.n_runhistory):
            median = sorted(ranking_loss_hist[:, i])[int(n_sampling * 0.5)]
            self.ignore_flag[i] = median > threshold
        print(self.ignore_flag)
        print('Updating weights took %.3f sec.' % (time.time() - _start_time))

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
        self.target_model = self.create_basic_model()
        self.target_model.train(X, y)

        self.is_trained = True
        if self.weight_update_id % self.n_steps_update == 0:
            self._update_weights(X, y)
        self.weight_update_id += 1

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
        if self.gp_fusion in ['indp-aspt', 'no-unct']:
            mu *= self.model_weights[-1]
            if self.gp_fusion == 'indp-aspt':
                var *= np.power(self.model_weights[-1], 2)

            # Base surrogate predictions with corresponding weights.
            for i in range(0, self.n_runhistory):
                if not self.ignore_flag[i]:
                    _w = self.model_weights[i]
                    _mu, _var = self.gp_models[i].predict(X_test)
                    mu += _w * _mu
                    if self.gp_fusion == 'indp-aspt':
                        var += _w * _w * _var
            return mu, var
        else:
            m = self.n_runhistory + 1
            ep = 1e-8
            mu_, var_ = np.zeros((n, m)), np.zeros((n, m))

            mu_t, var_t = mu.flatten(), var.flatten() + ep
            var_[:, -1] = 1. / var_t * self.model_weights[-1]
            mu_[:, -1] = 1. / var_t * mu_t * self.model_weights[-1]

            # Predictions from basic surrogates.
            for i in range(self.n_runhistory):
                mu_t, var_t = self.gp_models[i].predict(X_test)
                mu_t, var_t = mu_t.flatten(), var_t.flatten() + ep

                # compute the gaussian experts.
                var_[:, i] = 1. / var_t * self.model_weights[i]
                mu_[:, i] = 1. / var_t * mu_t * self.model_weights[i]

            var = 1. / np.sum(var_, axis=1)
            mu = np.sum(mu_, axis=1) * var
            assert np.isfinite(var).all()
            assert np.isfinite(mu).all()
            return mu.reshape((-1, 1)), var.reshape((-1, 1))
