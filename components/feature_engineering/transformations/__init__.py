import os
from components.utils.constants import FEATURE_TYPES
from components.utils.utils import find_components, collect_infos
from components.feature_engineering.transformations.base_transformer import Transformer

"""
Load the build-in transformers.
"""
transformers_directory = os.path.split(__file__)[0]
_transformers = find_components(__package__, transformers_directory, Transformer)

for sub_pkg in ['generator', 'preprocessor', 'rescaler', 'selector']:
    tmp_directory = os.path.split(__file__)[0] + '/%s' % sub_pkg
    transformers = find_components(__package__+'.%s' % sub_pkg, tmp_directory, Transformer)
    for key, val in transformers.items():
        if key not in _transformers:
            _transformers[key] = val
        else:
            raise ValueError('Repeated Transformer ID: %s!' % key)

_type_infos, _params_infos = collect_infos(_transformers, FEATURE_TYPES)
