import numpy as np
from solnml.components.optimizers.base.base_epm import AbstractEPM
from solnml.components.optimizers.base.prob_rf import RandomForestWithInstances


class WeightedRandomForestCluster(AbstractEPM):
    def __init__(self, types: np.ndarray,
                 bounds: np.ndarray, s_max, eta, weight_list, fusion_method, **kwargs):
        super().__init__(**kwargs)

        self.types = types
        self.bounds = bounds
        self.s_max = s_max
        self.eta = eta
        self.fusion = fusion_method
        self.surrogate_weight = dict()
        self.surrogate_container = dict()
        self.surrogate_r = list()
        self.weight_list = weight_list
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.surrogate_r.append(r)
            self.surrogate_weight[r] = self.weight_list[self.s_max - index]
            self.surrogate_container[r] = RandomForestWithInstances(types=types, bounds=bounds)

    def _train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        assert ('r' in kwargs)
        r = kwargs['r']
        self.surrogate_container[r].train(X, y)

    def _predict(self, X: np.ndarray):
        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.types.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (self.types.shape[0], X.shape[1]))
        if self.fusion == 'idp':
            means, vars = np.zeros((X.shape[0], 1)), np.zeros((X.shape[0], 1))
            for r in self.surrogate_r:
                mean, var = self.surrogate_container[r].predict(X)
                means += self.surrogate_weight[r] * mean
                vars += self.surrogate_weight[r] * self.surrogate_weight[r] * var
            return means.reshape((-1, 1)), vars.reshape((-1, 1))
        elif self.fusion == 'unct_ignore':
            means, vars = np.zeros((X.shape[0], 1)), np.zeros((X.shape[0], 1))
            for r in self.surrogate_r:
                mean, var = self.surrogate_container[r].predict(X)
                means += self.surrogate_weight[r] * mean
                if r == self.surrogate_r[-1]:
                    vars = var
            return means.reshape((-1, 1)), vars.reshape((-1, 1))
        elif self.fusion == 'gpoe':
            n = X.shape[0]
            m = len(self.surrogate_r)
            var_buf = np.zeros((n, m))
            mu_buf = np.zeros((n, m))
            # Predictions from base surrogates.
            for i, r in enumerate(self.surrogate_r):
                mu_t, var_t = self.surrogate_container[r].predict(X)
                mu_t = mu_t.flatten()
                var_t = var_t.flatten() + 1e-8
                # compute the gaussian experts.
                var_buf[:, i] = 1. / var_t * self.surrogate_weight[r]
                mu_buf[:, i] = 1. / var_t * mu_t * self.surrogate_weight[r]
            var = 1. / np.sum(var_buf, axis=1)
            mu = np.sum(mu_buf, axis=1) * var
            return mu.reshape((-1, 1)), var.reshape((-1, 1))
        else:
            raise ValueError('Undefined Fusion Method!')
