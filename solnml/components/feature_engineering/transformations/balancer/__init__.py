import os
from collections import OrderedDict
from solnml.components.feature_engineering.transformations.base_transformer import Transformer
from solnml.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
balancer_directory = os.path.split(__file__)[0]
_balancer = find_components(__package__, balancer_directory, Transformer)

_imb_balancer = OrderedDict()
# TODO:Verify the effect of smote_balancer
for key in ['weight_balancer', 'smote_balancer']:
    if key in _balancer.keys():
        _imb_balancer[key] = _balancer[key]

_bal_balancer = OrderedDict()
for key in ['weight_balancer']:
    if key in _balancer.keys():
        _bal_balancer[key] = _balancer[key]

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(Transformer)


def add_balancer(balancer):
    _addons.add_component(balancer)
