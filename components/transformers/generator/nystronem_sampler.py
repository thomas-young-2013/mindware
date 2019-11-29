from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition, EqualsCondition
from components.transformers.base_transformer import *


class NystronemSampler(Transformer):
    def __init__(self, kernel='rbf', n_components=100, gamma=1.0, degree=3,
                 coef0=1, random_state=None):
        super().__init__("nystronem_sampler", 15, random_state=random_state)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.kernel = kernel
        self.n_components = n_components
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data
        X_new = X[:, target_fields]

        if not self.model:
            import scipy
            from sklearn.kernel_approximation import Nystroem
            n_components = min(X.shape[0], self.n_components)

            self.gamma = float(self.gamma)
            self.degree = int(self.degree)
            self.coef0 = float(self.coef0)

            self.model = Nystroem(
                kernel=self.kernel, n_components=n_components,
                gamma=self.gamma, degree=self.degree, coef0=self.coef0,
                random_state=self.random_state)

            # Because the pipeline guarantees that each feature is positive,
            # clip all values below zero to zero
            if self.kernel == 'chi2':
                if scipy.sparse.issparse(X):
                    X.data[X.data < 0] = 0.0
                else:
                    X[X < 0] = 0.0

            self.model.fit(X_new.astype(np.float64))

        _X = self.model.transform(X_new)

        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        if dataset_properties is not None and \
                (dataset_properties.get("sparse") is True or
                 dataset_properties.get("signed") is False):
            allow_chi2 = False
        else:
            allow_chi2 = True

        possible_kernels = ['poly', 'rbf', 'sigmoid', 'cosine']
        if allow_chi2:
            possible_kernels.append("chi2")
        kernel = CategoricalHyperparameter('kernel', possible_kernels, 'rbf')
        n_components = UniformIntegerHyperparameter(
            "n_components", 10, 2000, default_value=100, log=True)
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                           log=True, default_value=0.1)
        degree = UniformIntegerHyperparameter('degree', 2, 5, 3)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([kernel, degree, gamma, coef0, n_components])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])

        gamma_kernels = ["poly", "rbf", "sigmoid"]
        if allow_chi2:
            gamma_kernels.append("chi2")
        gamma_condition = InCondition(gamma, kernel, gamma_kernels)
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])
        return cs
