import os
from automlToolkit.components.models.base_model import BaseClassificationModel
from automlToolkit.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin classifiers.
"""
classifiers_directory = os.path.split(__file__)[0]
_classifiers = find_components(__package__, classifiers_directory, BaseClassificationModel)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(BaseClassificationModel)


def add_classifier(classifier):
    _addons.add_component(classifier)
