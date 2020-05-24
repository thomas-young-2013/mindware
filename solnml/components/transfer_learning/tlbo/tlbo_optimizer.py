import abc
import logging
import numpy as np

from .utils.history_container import HistoryContainer
from .acquisition_function.acquisition import EI
from .models.gp import GaussianProcess
from litebo.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from litebo.optimizer.random_configuration_chooser import ChooserProb
from .utils.util_funcs import get_types, get_rng
from .config_space.util import convert_configurations_to_array
from .utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT
from .utils.limit import time_limit, TimeoutException


class BaseFacade(object, metaclass=abc.ABCMeta):
    def __init__(self, config_space, task_id):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.history_container = HistoryContainer(task_id)
        self.config_space = config_space

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def iterate(self):
        raise NotImplementedError()

    def get_history(self):
        return self.history_container

    def get_incumbent(self):
        return self.history_container.get_incumbents()


def create_gp_model(config_space, rng=None):
    from .models.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel
    from .models.gp_base_priors import LognormalPrior, HorseshoePrior

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


class TLBO(BaseFacade):
    def __init__(self, objective_function, config_space,
                 time_limit_per_trial=180,
                 max_runs=200,
                 initial_configurations=None,
                 past_runhistory=None,
                 initial_runs=3,
                 task_id=None,
                 rng=None):
        super().__init__(config_space, task_id)
        if rng is None:
            run_id, rng = get_rng()

        self.init_num = initial_runs
        self.max_iterations = max_runs
        self.iteration_id = 0
        self.sls_max_steps = None
        self.sls_n_steps_plateau_walk = 10
        self.time_limit_per_trial = time_limit_per_trial
        self.default_obj_value = MAXINT

        self.configurations = list()
        self.perfs = list()

        # Initialize the basic component in BO.
        self.objective_function = objective_function
        self.model = create_gp_model(config_space, rng)

        self.acquisition_function = EI(self.model)
        self.optimizer = InterleavedLocalAndRandomSearch(
                acquisition_function=self.acquisition_function,
                config_space=self.config_space,
                rng=np.random.RandomState(seed=rng.randint(MAXINT)),
                max_steps=self.sls_max_steps,
                n_steps_plateau_walk=self.sls_n_steps_plateau_walk
            )
        self._random_search = RandomSearch(
            self.acquisition_function, self.config_space, rng
        )
        self.random_configuration_chooser = ChooserProb(prob=0.5, rng=rng)

    def run(self):
        while self.iteration_id < self.max_iterations:
            self.iterate()

    def iterate(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            X = convert_configurations_to_array(self.configurations)
        Y = np.array(self.perfs, dtype=np.float64)
        config = self.choose_next(X, Y)

        trial_state = SUCCESS
        trial_info = None

        if config not in self.configurations:
            # Evaluate this configuration.
            try:
                with time_limit(self.time_limit_per_trial):
                    perf = self.objective_function(config)
            except Exception as e:
                perf = MAXINT
                trial_info = str(e)
                trial_state = FAILDED if not isinstance(e, TimeoutException) else TIMEOUT

            if len(self.configurations) == 0:
                self.default_obj_value = perf
            self.configurations.append(config)
            self.perfs.append(perf)
            self.history_container.add(config, perf)
        else:
            self.logger.debug('This configuration has been evaluated! Skip it.')
            config_idx = self.configurations.index(config)
            perf = self.perfs[config_idx]

        self.iteration_id += 1
        self.logger.debug('Iteration-%d, objective improvement: %.4f' % (self.iteration_id, max(0, self.default_obj_value - perf)))
        return config, trial_state, perf, trial_info

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        _config_num = X.shape[0]
        if _config_num < self.init_num:
            if _config_num == 0:
                return self.config_space.get_default_configuration()
            else:
                return self._random_search.maximize(runhistory=self.history_container, num_points=1)[0]

        self.model.train(X, Y)

        incumbent_value = self.history_container.get_incumbents()[0][1]

        self.acquisition_function.update(model=self.model, eta=incumbent_value, num_data=len(self.history_container.data))

        challengers = self.optimizer.maximize(
            runhistory=self.history_container,
            num_points=1000,
            random_configuration_chooser=self.random_configuration_chooser
        )
        return list(challengers)[0]
