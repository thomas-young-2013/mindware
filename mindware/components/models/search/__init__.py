import os
from mindware.components.models.base_nn import BaseImgClassificationNeuralNetwork
from mindware.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
searchers_directory = os.path.split(__file__)[0]
_searchers = find_components(__package__, searchers_directory, BaseImgClassificationNeuralNetwork)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(BaseImgClassificationNeuralNetwork)


def add_searcher(searcher):
    _addons.add_component(searcher)
