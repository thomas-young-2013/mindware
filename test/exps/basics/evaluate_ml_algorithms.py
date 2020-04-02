import os
import sys

sys.path.append(os.getcwd())

from automlToolkit.datasets.utils import load_train_test_data
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from autosklearn.pipeline.components.classification import _classifiers
from automlToolkit.components.evaluators.cls_evaluator import get_estimator


if __name__ == '__main__':
    classifier_id = 'lda'

    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", classifier_id)
    cs.add_hyperparameter(model)
    config_space = cs
    default_config = cs.get_default_configuration()
    config_space.seed(1)
    _, clf = get_estimator(default_config)

    raw_data, _ = load_train_test_data('yeast')
    X, y = raw_data.data
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(sum(y_pred == y))
