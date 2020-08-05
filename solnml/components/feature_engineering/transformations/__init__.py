import os
from solnml.components.utils.constants import FEATURE_TYPES
from solnml.components.utils.utils import find_components, collect_infos
from solnml.components.feature_engineering.transformations.base_transformer import Transformer
from solnml.components.feature_engineering.transformations.generator import _generator
from solnml.components.feature_engineering.transformations.selector import _selector
from solnml.components.feature_engineering.transformations.rescaler import _rescaler
from solnml.components.feature_engineering.transformations.preprocessor import _balancer
from solnml.components.feature_engineering.transformations.continous_discretizer import KBinsDiscretizer
from solnml.components.feature_engineering.transformations.discrete_categorizer import DiscreteCategorizer

"""
Load the build-in transformers.
"""
transformers_directory = os.path.split(__file__)[0]
_transformers = find_components(__package__, transformers_directory, Transformer)

for sub_pkg in ['generator', 'preprocessor', 'rescaler', 'selector']:
    tmp_directory = os.path.split(__file__)[0] + '/%s' % sub_pkg
    transformers = find_components(__package__ + '.%s' % sub_pkg, tmp_directory, Transformer)
    for key, val in transformers.items():
        if key not in _transformers:
            _transformers[key] = val
        else:
            raise ValueError('Repeated Transformer ID: %s!' % key)

_type_infos, _params_infos = collect_infos(_transformers, FEATURE_TYPES)

_preprocessor1 = {'continous_discretizer': KBinsDiscretizer}
_preprocessor2 = {'discrete_categorizer': DiscreteCategorizer}
_preprocessor = {}
for key in _generator:
    if key not in ['arithmetic_transformer', 'binary_transformer', 'lda_decomposer', 'pca_decomposer', 'kitchen_sinks']:
        _preprocessor[key] = _generator[key]
for key in _selector:
    if key not in ['rfe_selector', 'variance_selector', 'percentile_selector', 'percentile_selector_regression']:
        _preprocessor[key] = _selector[key]
