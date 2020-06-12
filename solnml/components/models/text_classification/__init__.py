import os
from solnml.components.models.base_nn import BaseTextClassificationNeuralNetwork
from solnml.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
classifiers_directory = os.path.split(__file__)[0]
_classifiers = find_components(__package__, classifiers_directory, BaseTextClassificationNeuralNetwork)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(BaseTextClassificationNeuralNetwork)


def add_classifier(classifier):
    _addons.add_component(classifier)