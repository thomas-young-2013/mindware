import os
from collections import OrderedDict
from solnml.components.utils.constants import FEATURE_TYPES
from solnml.components.utils.utils import find_components, collect_infos
from solnml.components.feature_engineering.transformations.base_transformer import Transformer
from solnml.components.feature_engineering.transformations.generator import _generator, _addons as _gen_addons, \
    add_generator
from solnml.components.feature_engineering.transformations.selector import _selector, _addons as _sel_addons, \
    add_selector
from solnml.components.feature_engineering.transformations.rescaler import _rescaler, _addons as _res_addons, \
    add_rescaler
from solnml.components.feature_engineering.transformations.balancer import _bal_balancer, _imb_balancer, \
    _addons as _bal_addons, add_balancer

from solnml.components.feature_engineering.transformations.preprocessor import _image_preprocessor, _text_preprocessor
from solnml.components.feature_engineering.transformations.empty_transformer import EmptyTransformer
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

_preprocessor1 = OrderedDict({'continous_discretizer': KBinsDiscretizer})
_preprocessor2 = OrderedDict({'discrete_categorizer': DiscreteCategorizer})
_preprocessor = OrderedDict()
for key in _generator:
    # if key not in ['arithmetic_transformer', 'binary_transformer', 'lda_decomposer', 'pca_decomposer', 'kitchen_sinks']:
    if key not in ['arithmetic_transformer', 'binary_transformer', 'lda_decomposer']:
        _preprocessor[key] = _generator[key]
for key in _selector:
    # if key not in ['rfe_selector', 'variance_selector', 'percentile_selector', 'percentile_selector_regression']:
    if key not in ['rfe_selector', 'variance_selector']:
        _preprocessor[key] = _selector[key]

_preprocessor1['empty'] = EmptyTransformer
_preprocessor1.move_to_end('empty', last=False)
_preprocessor2['empty'] = EmptyTransformer
_preprocessor2.move_to_end('empty', last=False)
_preprocessor['empty'] = EmptyTransformer
_preprocessor.move_to_end('empty', last=False)
_generator['empty'] = EmptyTransformer
_generator.move_to_end('empty', last=False)
_bal_balancer['empty'] = EmptyTransformer
_bal_balancer.move_to_end('empty', last=False)
_imb_balancer['empty'] = EmptyTransformer
_imb_balancer.move_to_end('empty', last=False)
_selector['empty'] = EmptyTransformer
_selector.move_to_end('empty', last=False)
_rescaler['empty'] = EmptyTransformer
_rescaler.move_to_end('empty', last=False)
