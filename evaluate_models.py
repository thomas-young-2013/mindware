from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UnParametrizedHyperparameter
from alphaml.engine.components.models.classification.adaboost import AdaboostClassifier

cs = AdaboostClassifier.get_hyperparameter_search_space()
model = UnParametrizedHyperparameter("estimator", 'random_forest')
cs.add_hyperparameter(model)

config = cs.get_default_configuration()
