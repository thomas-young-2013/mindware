import numpy as np
from fe_components.transformers.scaler import ScaleTransformation
from fe_components.transformers.binary_transformer import BinaryTransformation
from fe_components.transformers.generic_univariate_selector import GenericUnivariateSelector
from fe_components.transformation_graph import DataNode

from fe_components.utils.constants import *


def evaluate_feature_selectors():
    data = (np.array([
        [0, 1.2, 2, 1],
        [0, 1, 2, 1],
        [0, 3, 2, 2],
        [0, 5, 4, 5]
    ]), np.array([1, 2, 3, 4]))
    feature_type = [NUMERICAL, NUMERICAL, DISCRETE, DISCRETE]
    datanode = DataNode(data, feature_type)

    scaler = GenericUnivariateSelector(feature_left=3)
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    # transformer = VarianceSelector()
    # transformer = ModelBasedSelector(param='rf')
    # output_datanode = transformer.operate([datanode])
    print(output_datanode)
    print(output_datanode.data)
    print(output_datanode.feature_types)


if __name__ == '__main__':
    evaluate_feature_selectors()
