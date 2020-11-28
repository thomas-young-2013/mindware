import os
from solnml.components.feature_engineering.transformations.base_transformer import Transformer
from solnml.components.utils.class_loader import find_components

"""
Load the buildin classifiers.
"""
preprocessor_directory = os.path.split(__file__)[0]
_preprocessor = find_components(__package__, preprocessor_directory, Transformer)

_image_preprocessor = {}
_image_preprocessor['image2vector'] = _preprocessor['image2vector']

_text_preprocessor = {}
for key in ['text2vector', 'text2bertvector']:
    _text_preprocessor[key] = _preprocessor[key]
