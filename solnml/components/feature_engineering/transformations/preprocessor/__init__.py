import os
from solnml.components.feature_engineering.transformations.base_transformer import Transformer
from solnml.components.utils.class_loader import find_components

"""
Load the buildin classifiers.
"""
preprocessor_directory = os.path.split(__file__)[0]
_preprocessor = find_components(__package__, preprocessor_directory, Transformer)

_imb_balancer = {}
for key in ['weight_balancer', 'smote_balancer']:
    if key in _preprocessor.keys():
        _imb_balancer[key] = _preprocessor[key]

_bal_balancer = {}
for key in ['weight_balancer']:
    if key in _preprocessor.keys():
        _bal_balancer[key] = _preprocessor[key]
