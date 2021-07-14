import os
import shutil
import time
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mindware.utils.data_manager import DataManager
from mindware.estimators import Classifier
from mindware.components.models.base_model import BaseClassificationModel
from mindware.components.models.classification import add_classifier
from mindware.components.utils.configspace_utils import check_none
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class UserDefinedDecisionTree(BaseClassificationModel):
    def __init__(self, criterion, max_features, max_depth_factor,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_leaf_nodes, min_impurity_decrease, class_weight=None,
                 random_state=None):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth_factor = max_depth_factor
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weight = class_weight
        self.estimator = None
        self.time_limit = None

    def fit(self, X, y, sample_weight=None):
        from sklearn.tree import DecisionTreeClassifier

        self.max_features = float(self.max_features)
        # Heuristic to set the tree depth
        if check_none(self.max_depth_factor):
            max_depth_factor = self.max_depth_factor = None
        else:
            num_features = X.shape[1]
            self.max_depth_factor = int(self.max_depth_factor)
            max_depth_factor = max(
                1,
                int(np.round(self.max_depth_factor * num_features, 0)))
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.min_impurity_decrease = float(self.min_impurity_decrease)

        self.estimator = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=max_depth_factor,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            random_state=self.random_state)
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DT',
                'name': 'Decision Tree Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            criterion = CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default_value="gini")
            max_depth_factor = UniformFloatHyperparameter(
                'max_depth_factor', 0., 2., default_value=0.5)
            min_samples_split = UniformIntegerHyperparameter(
                "min_samples_split", 2, 20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                "min_samples_leaf", 1, 20, default_value=1)
            min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
            max_features = UnParametrizedHyperparameter('max_features', 1.0)
            max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
            min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

            cs.add_hyperparameters([criterion, max_features, max_depth_factor,
                                    min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, max_leaf_nodes,
                                    min_impurity_decrease])
            return cs


def main():
    time_limit = 60
    print('==> Start to evaluate with Budget %d' % time_limit)

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
    dm = DataManager(X_train, y_train)
    train_data = dm.get_data_node(X_train, y_train)
    test_data = dm.get_data_node(X_test, y_test)

    save_dir = './data/eval_exps/soln-ml'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    add_classifier(UserDefinedDecisionTree)
    clf = Classifier(time_limit=time_limit,
                     output_dir=save_dir,
                     enable_meta_algorithm_selection=False,
                     include_algorithms=['UserDefinedDecisionTree'],
                     ensemble_method=None,
                     metric='acc')
    _start_time = time.time()

    clf.fit(train_data)
    print(clf.summary())

    pred = clf.predict(test_data)
    print(accuracy_score(test_data.data[1], pred))

    shutil.rmtree(save_dir)


if __name__ == '__main__':
    main()
