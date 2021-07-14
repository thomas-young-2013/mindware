from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, Constant, CategoricalHyperparameter
from mindware.components.feature_engineering.transformations.base_transformer import *
from mindware.components.utils.configspace_utils import check_none, check_for_bool


class RandomTreesEmbeddingTransformation(Transformer):
    type = 18

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=1.0, max_leaf_nodes='None',
                 sparse_output=True, bootstrap='False', n_jobs=-1, random_state=1):
        super().__init__("random_trees_embedding")
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = CATEGORICAL

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.sparse_output = sparse_output
        self.n_jobs = n_jobs
        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode: DataNode, target_fields=None):
        from sklearn.ensemble import RandomTreesEmbedding

        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(input_datanode.feature_types, self.input_type)
        X_new = X[:, target_fields]
        if not self.model:
            self.n_estimators = int(self.n_estimators)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)

            # Skip heavy computation. max depth is set to 6.
            if X.shape[0] > 5000:
                self.max_depth = min(6, self.max_depth)

            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
            self.bootstrap = check_for_bool(self.bootstrap)

            self.model = RandomTreesEmbedding(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                sparse_output=self.sparse_output,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )

            self.model.fit(X_new)

        _X = self.model.transform(X_new).toarray()

        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        n_estimators = UniformIntegerHyperparameter(name="n_estimators",
                                                    lower=10, upper=100,
                                                    default_value=10)
        max_depth = UniformIntegerHyperparameter(name="max_depth",
                                                 lower=2, upper=10,
                                                 default_value=5)
        min_samples_split = UniformIntegerHyperparameter(name="min_samples_split",
                                                         lower=2, upper=20,
                                                         default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(name="min_samples_leaf",
                                                        lower=1, upper=20,
                                                        default_value=1)
        min_weight_fraction_leaf = Constant('min_weight_fraction_leaf', 1.0)
        max_leaf_nodes = UnParametrizedHyperparameter(name="max_leaf_nodes",
                                                      value="None")
        bootstrap = CategoricalHyperparameter('bootstrap', ['True', 'False'])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_estimators, max_depth, min_samples_split,
                                min_samples_leaf, min_weight_fraction_leaf,
                                max_leaf_nodes, bootstrap])
        return cs
