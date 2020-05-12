from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from solnml.components.feature_engineering.transformations.base_transformer import *
from solnml.components.utils.configspace_utils import check_for_bool, check_none


class LibLinearBasedSelector(Transformer):
    def __init__(self, penalty='l1', loss='squared_hinge', dual='False', tol=1e-4, C=1.0, multi_class='ovr',
                 fit_intercept='True', intercept_scaling=1, class_weight=None, random_state=1):
        super().__init__("liblinear_based_selector", 7)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'

        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.preprocessor = None

    def operate(self, input_datanode, target_fields=None):
        from sklearn.feature_selection import SelectFromModel

        feature_types = input_datanode.feature_types
        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(feature_types, self.input_type)
        X_new = X[:, target_fields]

        n_fields = len(feature_types)
        irrevalent_fields = list(range(n_fields))
        for field_id in target_fields:
            irrevalent_fields.remove(field_id)

        if self.model is None:
            from sklearn.svm import LinearSVC

            self.C = float(self.C)
            self.tol = float(self.tol)
            self.dual = check_for_bool(self.dual)
            self.fit_intercept = check_for_bool(self.fit_intercept)
            self.intercept_scaling = float(self.intercept_scaling)

            if check_none(self.class_weight):
                self.class_weight = None

            estimator = LinearSVC(penalty=self.penalty, loss=self.loss, dual=self.dual,
                                  tol=self.tol, C=self.C, class_weight=self.class_weight,
                                  fit_intercept=self.fit_intercept, intercept_scaling=self.intercept_scaling,
                                  multi_class=self.multi_class, random_state=self.random_state)

            estimator.fit(X_new, y)
            self.model = SelectFromModel(estimator, prefit=True, threshold='mean')

        _X = self.model.transform(X_new)
        is_selected = self.model.get_support()

        irrevalent_types = [feature_types[idx] for idx in irrevalent_fields]
        selected_types = [feature_types[idx] for idx in target_fields if is_selected[idx]]
        selected_types.extend(irrevalent_types)

        new_X = np.hstack((_X, X[:, irrevalent_fields]))
        new_feature_types = selected_types
        output_datanode = DataNode((new_X, y), new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)
        output_datanode.enable_balance = input_datanode.enable_balance
        output_datanode.data_balance = input_datanode.data_balance
        self.target_fields = target_fields.copy()

        return output_datanode

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        penalty = Constant("penalty", "l1")
        loss = CategoricalHyperparameter(
            "loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
        dual = Constant("dual", "False")
        # This is set ad-hoc
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        multi_class = Constant("multi_class", "ovr")
        # These are set ad-hoc
        fit_intercept = Constant("fit_intercept", "True")
        intercept_scaling = Constant("intercept_scaling", 1)

        cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class,
                                fit_intercept, intercept_scaling])

        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"),
            ForbiddenEqualsClause(loss, "hinge")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        return cs
