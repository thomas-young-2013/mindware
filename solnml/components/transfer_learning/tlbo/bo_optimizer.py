import abc
import logging
import numpy as np

from .acquisition_function.acquisition import EI
from .optimizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from .optimizer.random_configuration_chooser import ChooserProb
from .utils.util_funcs import get_types, get_rng
from .utils.history_container import HistoryContainer
from .utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT
from .utils.limit import time_limit, TimeoutException
from .config_space.util import convert_configurations_to_array
from .models.gp_ensemble import create_gp_model
from .models.rf_with_instances import RandomForestWithInstances


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


class BO(BaseFacade):
    def __init__(self, objective_function, config_space,
                 surrogate_model='gp',
                 time_limit_per_trial=180,
                 max_runs=200,
                 initial_configurations=None,
                 initial_runs=3,
                 task_id=None,
                 rng=None):
        super().__init__(config_space, task_id)
        if rng is None:
            run_id, rng = get_rng()

        self.surrogate_model = surrogate_model
        self.initial_configurations = initial_configurations
        self.init_num = initial_runs
        if initial_configurations is not None:
            self.init_num = len(initial_configurations)

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
        if self.surrogate_model == 'gp':
            self.model = create_gp_model(config_space, rng)
        elif self.surrogate_model == 'prob_rf':
            self.model = RandomForestWithInstances(config_space, seed=rng.randint(MAXINT), normalize_y=True)
        else:
            raise ValueError('Unsupported surrogate model - %s!' % self.surrogate_model)

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
        # Disable random configuration.
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
                print(self.iteration_id, str(e))

            if len(self.configurations) == 0:
                self.default_obj_value = perf
            self.configurations.append(config)
            self.perfs.append(perf)
            if trial_state == SUCCESS:
                self.history_container.add(config, perf)
        else:
            self.logger.debug('This configuration has been evaluated! Skip it.')
            config_idx = self.configurations.index(config)
            perf = self.perfs[config_idx]

        self.iteration_id += 1
        print(self.iteration_id)
        self.logger.debug('Iteration-%d, objective improvement: %.4f' % (self.iteration_id, max(0, self.default_obj_value - perf)))
        return config, trial_state, perf, trial_info

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        _config_num = X.shape[0]
        if _config_num < self.init_num:
            if self.initial_configurations is None:
                if _config_num == 0:
                    return self.config_space.get_default_configuration()
                else:
                    return self._random_search.maximize(runhistory=self.history_container, num_points=1)[0]
            else:
                return self.initial_configurations[_config_num]

        self.model.train(X, Y)

        incumbent_value = self.history_container.get_incumbents()[0][1]

        self.acquisition_function.update(model=self.model, eta=incumbent_value, num_data=len(self.history_container.data))

        challengers = self.optimizer.maximize(
            runhistory=self.history_container,
            num_points=1000,
            random_configuration_chooser=self.random_configuration_chooser
        )
        config = list(challengers)[0]
        assert config.origin != 'Random Search'
        return config
