import numpy as np
import sklearn.gaussian_process.kernels
from typing import List, Optional, Tuple, Union
from .gp_base_priors import Prior, SoftTopHatPrior, TophatPrior
from .abstract_model import AbstractModel


class BaseModel(AbstractModel):
    def __init__(self, configspace, types, bounds, seed, **kwargs):
        """
        Abstract base class for all Gaussian process models.
        """
        super().__init__(configspace=configspace, types=types, bounds=bounds, seed=seed, **kwargs)

        self.rng = np.random.RandomState(seed)

    def _get_all_priors(
        self,
        add_bound_priors: bool = True,
        add_soft_bounds: bool = False,
    ) -> List[List[Prior]]:
        # Obtain a list of all priors for each tunable hyperparameter of the kernel
        all_priors = []
        to_visit = []
        to_visit.append(self.gp.kernel.k1)
        to_visit.append(self.gp.kernel.k2)
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                continue
            elif isinstance(current_param, sklearn.gaussian_process.kernels.Kernel):
                hps = current_param.hyperparameters
                assert len(hps) == 1
                hp = hps[0]
                if hp.fixed:
                    continue
                bounds = hps[0].bounds
                for i in range(hps[0].n_elements):
                    priors_for_hp = []
                    if current_param.prior is not None:
                        priors_for_hp.append(current_param.prior)
                    if add_bound_priors:
                        if add_soft_bounds:
                            priors_for_hp.append(SoftTopHatPrior(
                               lower_bound=bounds[i][0], upper_bound=bounds[i][1], rng=self.rng,
                            ))
                        else:
                            priors_for_hp.append(TophatPrior(
                                lower_bound=bounds[i][0], upper_bound=bounds[i][1], rng=self.rng,
                            ))
                    all_priors.append(priors_for_hp)
        return all_priors

    def _set_has_conditions(self):
        has_conditions = len(self.configspace.get_conditions()) > 0
        to_visit = []
        to_visit.append(self.kernel)
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                current_param.has_conditions = has_conditions
            elif isinstance(current_param, sklearn.gaussian_process.kernels.Kernel):
                current_param.has_conditions = has_conditions
            else:
                raise ValueError(current_param)

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[~np.isfinite(X)] = -1
        return X
