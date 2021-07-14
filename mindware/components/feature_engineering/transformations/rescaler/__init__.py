import os
from mindware.components.feature_engineering.transformations.base_transformer import Transformer
from mindware.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
rescaler_directory = os.path.split(__file__)[0]
_rescaler = find_components(__package__, rescaler_directory, Transformer)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(Transformer)


def add_rescaler(rescaler):
    _addons.add_component(rescaler)
