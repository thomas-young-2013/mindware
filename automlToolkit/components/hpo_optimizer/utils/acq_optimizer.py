import numpy as np
from ConfigSpace.util import get_one_exchange_neighbourhood

from automlToolkit.components.hpo_optimizer.utils.config_space_utils import convert_configurations_to_array, \
    sample_configurations


class BaseOptimizer(object):

    def __init__(self, objective_function, config_space, rng=None):
        """
        Interface for optimizers that maximizing the
        acquisition function.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        rng: numpy.random.RandomState
            Random number generator
        """
        self.config_space = config_space
        self.objective_func = objective_function
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(10000))
        else:
            self.rng = rng

    def maximize(self):
        pass


class RandomSampling(BaseOptimizer):

    def __init__(self, objective_function, config_space, n_samples=500, rng=None):
        """
        Samples candidates uniformly at random and returns the point with the highest objective value.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_samples: int
            Number of candidates that are samples
        """
        self.n_samples = n_samples
        super(RandomSampling, self).__init__(objective_function, config_space, rng)

    def maximize(self, batch_size=1):
        """
        Maximizes the given acquisition function.

        Parameters
        ----------
        batch_size: number of maximizer returned.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """
        eta = 0.3
        incs_num = int(eta * self.n_samples)
        incs_configs = list(
            get_one_exchange_neighbourhood(self.objective_func.eta['config'], seed=self.rng.randint(int(1e6))))
        # TODO: need to implement
        # extra_num = incs_num - len(incs_configs)
        # if extra_num > 0:
        #     incs_configs.extend(get_random_neighborhood(self.objective_func.eta['config'], extra_num, MAXINT))

        configs_list = list(incs_configs)
        rand_incs = convert_configurations_to_array(configs_list)

        # Sample random points uniformly over the whole space
        # rand_configs = self.config_space.sample_configuration(self.n_samples - rand_incs.shape[0])
        rand_configs = sample_configurations(self.config_space, self.n_samples - rand_incs.shape[0])
        rand = convert_configurations_to_array(rand_configs)

        configs_list.extend(rand_configs)

        # TODO: Put a Gaussian on the incumbent and sample from that (support categorical feature)
        # loc = self.objective_func.model.get_incumbent()[0],
        # scale = np.ones([self.lower.shape[0]]) * 0.1
        # rand_incs = np.array([np.clip(np.random.normal(loc, scale), self.lower, self.upper)[0]
        #                       for _ in range(int(self.n_samples * 0.3))])
        #

        X = np.concatenate((rand_incs, rand), axis=0)
        y = self.objective_func(X)
        if batch_size == 1:
            return [configs_list[np.argmax(y)]]

        tmp = configs_list[np.argsort(y)[-batch_size:]]
        return tmp
