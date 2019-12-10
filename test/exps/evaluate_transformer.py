import os
import sys
import numpy as np
sys.path.append(os.getcwd())
from automlToolkit.components.feature_engineering.transformations.selector.generic_univariate_selector import GenericUnivariateSelector
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.utils.constants import *


def evaluate_feature_selectors():
    data = (np.array([
        [0, 1.2, 2, 1],
        [0, 1, 2, 1],
        [0, 3, 2, 2],
        [0, 5, 4, 5]
    ]), np.array([1, 2, 3, 4]))
    feature_type = [NUMERICAL, NUMERICAL, DISCRETE, DISCRETE]
    datanode = DataNode(data, feature_type)

    scaler = GenericUnivariateSelector()
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    # transformer = VarianceSelector()
    # transformer = ModelBasedSelector(param='rf')
    # output_datanode = transformer.operate([datanode])
    print(output_datanode)
    print(output_datanode.data)
    print(output_datanode.feature_types)


def test_selector():
    data = (np.array([
        [0, 1.2, 2, 1],
        [0, 1, 2, 1],
        [0, 3, 2, 2],
        [0, 5, 4, 5]
    ]), np.array([1, 2, 3, 4]))
    feature_type = [NUMERICAL, NUMERICAL, DISCRETE, DISCRETE]
    datanode = DataNode(data, feature_type)

    # Test generic univariate selector.
    scaler = GenericUnivariateSelector()
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test percentile selector.
    from automlToolkit.components.feature_engineering.transformations.selector.percentile_selector import PercentileSelector
    scaler = PercentileSelector(percentile=25)
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test model based selector.
    from automlToolkit.components.feature_engineering.transformations.selector.model_based_selector import ModelBasedSelector
    scaler = ModelBasedSelector(param='et')
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test variance threshold.
    from automlToolkit.components.feature_engineering.transformations.selector.variance_selector import VarianceSelector
    scaler = VarianceSelector()
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)


def test_additional_transformations():
    data = (np.array([
        [0, 1.2, 2, 1],
        [0.01, 1, 2, 1],
        [0.02, 3, 2, 2],
        [0.015, 5, 4, 5],
        [0.12, 3, 2, 2],
        [0.16, 5, 4, 5]
    ]), np.array([1, 1, 2, 2, 3, 3]))
    feature_type = [NUMERICAL, NUMERICAL, DISCRETE, DISCRETE]
    datanode = DataNode(data, feature_type)
    from automlToolkit.components.feature_engineering.transformations.generator.arithmetic_transformer import ArithmeticTransformation
    from automlToolkit.components.feature_engineering.transformations.generator.lda_decomposer import LdaDecomposer
    from automlToolkit.components.feature_engineering.transformations.continous_discretizer import KBinsDiscretizer
    from automlToolkit.components.feature_engineering.transformations.discrete_categorizer import DiscreteCategorizer
    # trans = ArithmeticTransformation()
    # trans = LdaDecomposer()
    # trans = KBinsDiscretizer()
    trans = DiscreteCategorizer()
    output_datanode = trans.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)


def test_generator():
    data = (np.array([
        [0, 1.2, 2, 1],
        [0, 1, 2, 1],
        [0, 3, 2, 2],
        [0, 5, 4, 5]
    ]), np.array([1, 2, 3, 4]))
    feature_type = [NUMERICAL, NUMERICAL, DISCRETE, DISCRETE]
    datanode = DataNode(data, feature_type)

    # Test SVD.
    from automlToolkit.components.feature_engineering.transformations.generator.svd_decomposer import SvdDecomposer
    scaler = SvdDecomposer()
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test feature agglomerate.
    from automlToolkit.components.feature_engineering.transformations.generator.feature_agglomeration_decomposer import FeatureAgglomerationDecomposer
    scaler = FeatureAgglomerationDecomposer()
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test PCA.
    from automlToolkit.components.feature_engineering.transformations.generator.pca_decomposer import PcaDecomposer
    scaler = PcaDecomposer()
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test kernel PCA.
    from automlToolkit.components.feature_engineering.transformations.generator.kernel_pca import KernelPCA
    scaler = KernelPCA()
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test fast ICA.
    from automlToolkit.components.feature_engineering.transformations.generator.fast_ica_decomposer import FastIcaDecomposer
    scaler = FastIcaDecomposer()
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test LDA.
    # from components.transformers.generator.lda_decomposer import LdaDecomposer
    # scaler = LdaDecomposer(frac=0.3)
    # scaler.concatenate = False
    # output_datanode = scaler.operate(datanode)
    # print(output_datanode)
    # print(output_datanode.data)

    # Test random trees embedding.
    from automlToolkit.components.feature_engineering.transformations.generator.random_trees_embedding import RandomTreesEmbeddingTransformation
    scaler = RandomTreesEmbeddingTransformation()
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)


if __name__ == '__main__':
    # test_selector()
    # test_generator()
    test_additional_transformations()
