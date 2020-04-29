import os
from automlToolkit.components.models.base_model import BaseClassificationModel
from automlToolkit.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
unbalanced_classifiers_directory = os.path.split(__file__)[0]
_ubl_classifiers = find_components(__package__, unbalanced_classifiers_directory, BaseClassificationModel)
