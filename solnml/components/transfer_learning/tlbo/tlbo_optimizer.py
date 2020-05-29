import numpy as np

from .acquisition_function.acquisition import EI
from .optimizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from .optimizer.random_configuration_chooser import ChooserProb
from .utils.util_funcs import get_types, get_rng
from .utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT
from .utils.limit import time_limit, TimeoutException
from .config_space.util import convert_configurations_to_array
from .bo_optimizer import BaseFacade
from .models.gp_ensemble import GaussianProcessEnsemble


class TLBO(BaseFacade):
    def __init__(self, objective_function,
                 config_space,
                 past_runhistory,
                 dataset_metafeature=None,
                 time_limit_per_trial=180,
                 max_runs=200,
                 initial_runs=5,
                 task_id=None,
                 rng=None):
        super().__init__(config_space, task_id)
        if rng is None:
            _, rng = get_rng()
        self.rng = rng
        self.past_runhistory = past_runhistory
        self.meta_feature_scaler=None
        self.dataset_metafeature = dataset_metafeature
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
        seed = rng.randint(MAXINT)
        self.model = GaussianProcessEnsemble(past_runhistory, config_space, seed=seed)
        self.initial_configurations = self.get_initial_configs()

        self.acquisition_function = EI(self.model)
        self.optimizer = InterleavedLocalAndRandomSearch(
                acquisition_function=self.acquisition_function,
                config_space=self.config_space,
                rng=np.random.RandomState(seed=seed),
                max_steps=self.sls_max_steps,
                n_steps_plateau_walk=self.sls_n_steps_plateau_walk
            )
        self._random_search = RandomSearch(
            self.acquisition_function, self.config_space, rng
        )
        self.random_configuration_chooser = ChooserProb(prob=0.5, rng=rng)

    def get_initial_configs(self):
        """
            runhistory format:
                row: [ dataset_metafeature, list([[configuration, perf],[]]) ]
        """
        from sklearn.preprocessing import MinMaxScaler
        meta_features = list()
        for _runhistory in self.past_runhistory:
            meta_features.append(_runhistory[0])
        meta_features = np.array(meta_features)
        self.meta_feature_scaler = MinMaxScaler()
        meta_features = self.meta_feature_scaler.fit_transform(meta_features)

        init_configs = [self.config_space.get_default_configuration()]
        n_init_configs = self.init_num - 1
        if self.dataset_metafeature is not None:
            dataset_metafeature = self.meta_feature_scaler.transform(self.dataset_metafeature.reshape((1, -1)))
            euclidean_distance = list()
            for _metafeeature in meta_features:
                euclidean_distance.append(np.linalg.norm(dataset_metafeature - _metafeeature))
            history_idxs = np.argsort(euclidean_distance)[:n_init_configs]
        else:
            idxs = np.arange(len(self.past_runhistory))
            np.random.shuffle(idxs)
            history_idxs = idxs[:n_init_configs]

        for _idx in history_idxs:
            config_perf_pairs = self.past_runhistory[_idx][1]
            perfs = [row[1] for row in config_perf_pairs]
            optimum_idx = np.argsort(perfs)[0]
            init_configs.append(config_perf_pairs[optimum_idx][0])
        return list(set(init_configs))

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

        self.acquisition_function.update(model=self.model, eta=incumbent_value,
                                         num_data=len(self.history_container.data))

        challengers = self.optimizer.maximize(
            runhistory=self.history_container,
            num_points=1000,
            random_configuration_chooser=self.random_configuration_chooser
        )
        return list(challengers)[0]
