import os
from solnml.components.feature_engineering.transformations.base_transformer import Transformer
from solnml.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
selector_directory = os.path.split(__file__)[0]
_selector = find_components(__package__, selector_directory, Transformer)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(Transformer)


def add_selector(selector):
    _addons.add_component(selector)
