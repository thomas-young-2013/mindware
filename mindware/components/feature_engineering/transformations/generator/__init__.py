import os
from mindware.components.feature_engineering.transformations.base_transformer import Transformer
from mindware.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
generator_directory = os.path.split(__file__)[0]
_generator = find_components(__package__, generator_directory, Transformer)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(Transformer)


def add_generator(generator):
    _addons.add_component(generator)
