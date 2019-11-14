import os
from fe_components.utils.constants import FEATURE_TYPES
from fe_components.utils.utils import find_components, collect_infos
from fe_components.transformers.base_transformer import Transformer

"""
Load the build-in transformers.
"""
transformers_directory = os.path.split(__file__)[0]
_transformers = find_components(__package__, transformers_directory, Transformer)
_type_infos, _params_infos = collect_infos(_transformers, FEATURE_TYPES)
