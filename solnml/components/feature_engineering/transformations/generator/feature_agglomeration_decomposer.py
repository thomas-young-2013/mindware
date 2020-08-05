from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenInClause, \
    ForbiddenAndConjunction, ForbiddenEqualsClause
from solnml.components.feature_engineering.transformations.base_transformer import *


class FeatureAgglomerationDecomposer(Transformer):
    def __init__(self, n_clusters=2, affinity='euclidean', linkage='ward', pooling_func='mean',
                 random_state=1):
        super().__init__("feature_agglomeration_decomposer", 11)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func = pooling_func
        self.random_state = random_state

        self.pooling_func_mapping = dict(mean=np.mean, median=np.median, max=np.max)

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from sklearn.cluster import FeatureAgglomeration

        X, y = input_datanode.data

        if self.model is None:
            self.n_clusters = int(self.n_clusters)

            n_clusters = min(self.n_clusters, X.shape[1])
            if not callable(self.pooling_func):
                self.pooling_func = self.pooling_func_mapping[self.pooling_func]

            self.model = FeatureAgglomeration(
                n_clusters=n_clusters, affinity=self.affinity,
                linkage=self.linkage, pooling_func=self.pooling_func)
            self.model.fit(X)

        X_new = self.model.transform(X)

        return X_new

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        cs = ConfigurationSpace()
        n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 400, default_value=25)
        affinity = CategoricalHyperparameter(
            "affinity", ["euclidean", "manhattan", "cosine"], default_value="euclidean")
        linkage = CategoricalHyperparameter(
            "linkage", ["ward", "complete", "average"], default_value="ward")
        pooling_func = CategoricalHyperparameter(
            "pooling_func", ["mean", "median", "max"], default_value="mean")

        cs.add_hyperparameters([n_clusters, affinity, linkage, pooling_func])

        affinity_and_linkage = ForbiddenAndConjunction(
            ForbiddenInClause(affinity, ["manhattan", "cosine"]),
            ForbiddenEqualsClause(linkage, "ward"))
        cs.add_forbidden_clause(affinity_and_linkage)
        return cs
