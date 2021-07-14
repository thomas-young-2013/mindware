import os
from mindware.components.models.base_model import BaseClassificationModel
from mindware.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
imbalanced_classifiers_directory = os.path.split(__file__)[0]
_imb_classifiers = find_components(__package__, imbalanced_classifiers_directory, BaseClassificationModel)
