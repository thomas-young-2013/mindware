import numpy as np
from components.transformers.selector.generic_univariate_selector import GenericUnivariateSelector
from components.transformation_graph import DataNode

from components.utils.constants import *


def evaluate_feature_selectors():
    data = (np.array([
        [0, 1.2, 2, 1],
        [0, 1, 2, 1],
        [0, 3, 2, 2],
        [0, 5, 4, 5]
    ]), np.array([1, 2, 3, 4]))
    feature_type = [NUMERICAL, NUMERICAL, DISCRETE, DISCRETE]
    datanode = DataNode(data, feature_type)

    scaler = GenericUnivariateSelector(feature_left=0.5)
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
    scaler = GenericUnivariateSelector(feature_left=0.5)
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test percentile selector.
    from components.transformers.selector.percentile_selector import PercentileSelector
    scaler = PercentileSelector(percentile=25)
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test model based selector.
    from components.transformers.selector.model_based_selector import ModelBasedSelector
    scaler = ModelBasedSelector(param='et')
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test variance threshold.
    from components.transformers.selector.variance_selector import VarianceSelector
    scaler = VarianceSelector()
    output_datanode = scaler.operate(datanode)
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
    from components.transformers.generator.svd_decomposer import SvdDecomposer
    scaler = SvdDecomposer(frac=0.5)
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test feature agglomerate.
    from components.transformers.generator.feature_agglomeration_decomposer import FeatureAgglomerationDecomposer
    scaler = FeatureAgglomerationDecomposer(frac=0.5)
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test PCA.
    from components.transformers.generator.pca_decomposer import PcaDecomposer
    scaler = PcaDecomposer(frac=0.99)
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test kernel PCA.
    from components.transformers.generator.kernel_pca import KernelPCA
    scaler = KernelPCA()
    scaler.concatenate = False
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)

    # Test fast ICA.
    from components.transformers.generator.fast_ica_decomposer import FastIcaDecomposer
    scaler = FastIcaDecomposer(frac=0.5)
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
    from components.transformers.generator.random_trees_embedding import RandomTreesEmbeddingTransformation
    scaler = RandomTreesEmbeddingTransformation()
    output_datanode = scaler.operate(datanode)
    print(output_datanode)
    print(output_datanode.data)


if __name__ == '__main__':
    # test_selector()
    test_generator()
