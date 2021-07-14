import os
from mindware.components.models.base_model import BaseRegressionModel
from mindware.components.utils.class_loader import find_components, ThirdPartyComponents

"""
Load the buildin regressors.
"""
regressors_directory = os.path.split(__file__)[0]
_regressors = find_components(__package__, regressors_directory, BaseRegressionModel)

"""
Load third-party classifiers. 
"""
_addons = ThirdPartyComponents(BaseRegressionModel)


def add_regressor(regressor):
    _addons.add_component(regressor)
