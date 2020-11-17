import os
from solnml.components.feature_engineering.transformations.base_transformer import Transformer
from solnml.components.utils.class_loader import find_components

"""
Load the buildin classifiers.
"""
preprocessor_directory = os.path.split(__file__)[0]
_preprocessor = find_components(__package__, preprocessor_directory, Transformer)

_imb_balancer = {}
# TODO:Verify the effect of smote_balancer
# for key in ['weight_balancer', 'smote_balancer']:
for key in ['weight_balancer']:
    if key in _preprocessor.keys():
        _imb_balancer[key] = _preprocessor[key]

_bal_balancer = {}
for key in ['weight_balancer']:
    if key in _preprocessor.keys():
        _bal_balancer[key] = _preprocessor[key]

_image_preprocessor = {}
_image_preprocessor['image2vector'] = _preprocessor['image2vector']

_text_preprocessor = {}
_text_preprocessor['text2vector'] = _preprocessor['text2vector']
