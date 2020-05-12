import os
from solnml.components.feature_engineering.transformations.base_transformer import Transformer
from solnml.components.utils.class_loader import find_components

"""
Load the buildin classifiers.
"""
rescaler_directory = os.path.split(__file__)[0]
_rescaler = find_components(__package__, rescaler_directory, Transformer)
