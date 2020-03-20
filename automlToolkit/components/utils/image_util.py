from keras.models import Model
import numpy as np
import warnings

try:
    from PIL import Image
except ImportError:
    warnings.warn("Pillow not installed! Image2Vector will fail!")


def reshape(images):
    reshaped_array = np.zeros((len(images), 224, 224, 3))
    for i in range(len(images)):
        image = Image.fromarray(images[i].astype('uint8'))
        reshaped_image = image.resize((224, 224))
        reshaped_array[i, :, :, :] = reshaped_image
    return reshaped_array


class Image2vector():
    def __init__(self, model='resnet'):
        if model == 'resnet':
            from keras.applications import ResNet50
            self.model = ResNet50(include_top=True)
        elif model == 'vgg':
            from keras.applications import VGG19
            self.model = VGG19(include_top=True)
        self.model = Model(inputs=self.model.input, outputs=self.model.output)

    def predict(self, images):
        """
        :param images: numpy array
        :return: numpy array of shape (n_samples,1000)
        """
        reshaped_images = reshape(images)
        return self.model.predict(reshaped_images, batch_size=128)
