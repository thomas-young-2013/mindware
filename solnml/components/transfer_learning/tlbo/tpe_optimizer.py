import numpy as np

from .bo_optimizer import BaseFacade
from .utils.util_funcs import get_rng
from .utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT
from .utils.limit import time_limit, TimeoutException
from .config_space.util import convert_configurations_to_array
from solnml.components.transfer_learning.tlbo.models.kde import TPE
from solnml.components.utils.mfse_utils.config_space_utils import sample_configurations


class TPE_BO(BaseFacade):
    def __init__(self, objective_function, config_space,
                 surrogate_model='tpe',
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
        self.time_limit_per_trial = time_limit_per_trial
        self.default_obj_value = MAXINT

        self.configurations = list()
        self.failed_configurations = list()
        self.perfs = list()

        # Initialize the basic component in BO.
        self.objective_function = objective_function
        if self.surrogate_model == 'tpe':
            self.model = TPE(configspace=config_space)
        elif self.surrogate_model == 'random_search':
            pass
        else:
            raise ValueError('Unsupported surrogate model - %s!' % self.surrogate_model)

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

        if config not in (self.configurations + self.failed_configurations):
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

            if trial_state == SUCCESS and perf < MAXINT:
                self.configurations.append(config)
                self.perfs.append(perf)
                # Update KDE model.
                if self.surrogate_model == 'tpe':
                    self.model.new_result(config, perf, 1)
                self.history_container.add(config, perf)
            else:
                self.failed_configurations.append(config)
        else:
            self.logger.debug('This configuration has been evaluated! Skip it.')
            if config in self.configurations:
                config_idx = self.configurations.index(config)
                trial_state, perf = SUCCESS, self.perfs[config_idx]
            else:
                trial_state, perf = FAILDED, MAXINT

        self.iteration_id += 1
        self.logger.debug('Iteration-%d, objective improvement: %.4f' % (self.iteration_id, max(0, self.default_obj_value - perf)))
        return config, trial_state, perf, trial_info

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        _config_num = X.shape[0]
        if _config_num < self.init_num:
            if self.initial_configurations is None:
                if _config_num == 0:
                    return self.config_space.get_default_configuration()
                else:
                    return sample_configurations(self.config_space, 1)[0]
            else:
                return self.initial_configurations[_config_num]
        else:
            if self.surrogate_model == 'tpe':
                config, _ = self.model.get_config()
            elif self.surrogate_model == 'random_search':
                config = sample_configurations(self.config_space, 1)[0]
            else:
                 raise ValueError('Invalid surrogate model - %s.' % self.surrogate_model)
            return config
