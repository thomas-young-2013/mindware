import numpy as np
import warnings
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from solnml.components.feature_engineering.transformations.base_transformer import *
from solnml.components.utils.image_util import Image2vector


class Image2VectorTransformation(Transformer):
    type = 51

    def __init__(self, method='resnet'):
        super().__init__("image2vector")
        self.method = method
        self.input_type = [IMAGE]
        self.output_type = [IMAGE_EMBEDDING]
        self.compound_mode = 'replace'
        self.pretrained_model = None

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data
        X_new = X[:, target_fields]
        if not self.pretrained_model:
            self.pretrained_model = Image2vector(model=self.method)
        _X = None
        for i in range(X_new.shape[1]):
            images = np.array([image for image in X_new[:, i]])
            emb_output = self.pretrained_model.predict(images)
            if _X is None:
                _X = emb_output.copy()
            else:
                _X = np.hstack((_X, emb_output))
        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        method = CategoricalHyperparameter("method", ['resnet', 'vgg'], default_value='resnet')

        cs = ConfigurationSpace()
        cs.add_hyperparameters([method])

        return cs
