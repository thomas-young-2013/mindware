import itertools
from fe_components.transformers.base_transformer import *

log_space = lambda x1, x2, num: np.exp(np.linspace(np.log(x1), np.log(x2), num))
lin_space = lambda x1, x2, num: np.linspace(x1, x2, num)


class KitchenSinks(Transformer):
    def __init__(self, param=(1, 100), seed=1):
        super().__init__("nystronem_sampler", 9, random_state=seed)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.output_type = NUMERICAL
        self.params = {'gamma': param[0], 'n_components': param[1]}
        gammas = log_space(0.001, 8., 7)
        # gammas = [3]
        n_components = log_space(50, 1000, 7)
        # n_components = [900]
        options = list(itertools.product(gammas, n_components))
        optional_params = list()
        for idx in np.random.choice(len(options), self.sample_size):
            optional_params.append(options[idx])
        self.optional_params = optional_params

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data
        X_new = X[:, target_fields]

        if not self.model:
            import sklearn.kernel_approximation
            self.model = sklearn.kernel_approximation.RBFSampler(
                gamma=self.params['gamma'], n_components=int(self.params['n_components']), random_state=self.random_state)
            self.model.fit(X_new)

        _X = self.model.transform(X_new)

        return _X
