import warnings
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from automlToolkit.components.feature_engineering.transformations.base_transformer import *


class KernelPCA(Transformer):
    def __init__(self, n_components=0.3, kernel='rbf', degree=3, gamma=0.25, coef0=0.0,
                 random_state=None):
        super().__init__("kernel_pca", 12)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state
        self.skip_flag = False
        self.pre_trained = False

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data

        # Skip large matrix computation in obtaining the kernel matrix.
        if X.shape[0] > 10000:
            if not self.pre_trained:
                self.skip_flag = True

        self.pre_trained = True
        if self.skip_flag:
            return X.copy()

        if self.model is None:
            import scipy.sparse
            from automlToolkit.components.feature_engineering.transformations.utils import KernelPCA

            self.n_components = int(self.n_components)
            self.degree = int(self.degree)
            self.gamma = float(self.gamma)
            self.coef0 = float(self.coef0)

            self.model = KernelPCA(
                n_components=self.n_components, kernel=self.kernel,
                degree=self.degree, gamma=self.gamma, coef0=self.coef0,
                remove_zero_eig=True, random_state=self.random_state)
            if scipy.sparse.issparse(X):
                X = X.astype(np.float64)
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                self.model.fit(X)
            # Raise an informative error message, equation is based ~line 249 in
            # kernel_pca.py in scikit-learn
            if len(self.model.alphas_ / self.model.lambdas_) == 0:
                raise ValueError('KernelPCA removed all features!')
        X_new = self.model.transform(X)

        return X_new

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_components = UniformIntegerHyperparameter(
            "n_components", 10, 2000, default_value=100)
        kernel = CategoricalHyperparameter('kernel',
                                           ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf')
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                           log=True, default_value=1.0)
        degree = UniformIntegerHyperparameter('degree', 2, 5, 3)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components, kernel, degree, gamma, coef0])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["poly", "rbf"])
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])
        return cs
