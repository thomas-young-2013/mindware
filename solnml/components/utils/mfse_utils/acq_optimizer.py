import numpy as np
from ConfigSpace.util import get_one_exchange_neighbourhood

from solnml.components.utils.mfse_utils.config_space_utils import convert_configurations_to_array, \
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

        incs_configs = list(
            get_one_exchange_neighbourhood(self.objective_func.eta['config'], seed=self.rng.randint(int(1e6))))

        configs_list = list(incs_configs)
        rand_incs = convert_configurations_to_array(configs_list)

        # Sample random points uniformly over the whole space
        rand_configs = sample_configurations(self.config_space, self.n_samples - rand_incs.shape[0])
        rand = convert_configurations_to_array(rand_configs)

        configs_list.extend(rand_configs)

        X = np.concatenate((rand_incs, rand), axis=0)
        y = self.objective_func(X).flatten()
        candidate_idxs = list(np.argsort(-y)[:batch_size])
        # print(candidate_idxs)
        # print(type(candidate_idxs))
        # print(configs_list[:5])
        return [configs_list[idx] for idx in candidate_idxs]
