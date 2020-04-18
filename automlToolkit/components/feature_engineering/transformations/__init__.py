import os
from automlToolkit.components.utils.constants import FEATURE_TYPES
from automlToolkit.components.utils.utils import find_components, collect_infos
from automlToolkit.components.feature_engineering.transformations.base_transformer import Transformer
from automlToolkit.components.feature_engineering.transformations.generator import _generator
from automlToolkit.components.feature_engineering.transformations.selector import _selector
from automlToolkit.components.feature_engineering.transformations.rescaler import _rescaler
from automlToolkit.components.feature_engineering.transformations.preprocessor import _balancer
from automlToolkit.components.feature_engineering.transformations.continous_discretizer import KBinsDiscretizer
from automlToolkit.components.feature_engineering.transformations.discrete_categorizer import DiscreteCategorizer

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
