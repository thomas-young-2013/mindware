"""
Transforms code borrowed from
https://github.com/automl/Auto-PyTorch/blob/fcc7de1b6072616daceac986389a0216451f4075/autoPyTorch/components/preprocessing/image_preprocessing/transforms.py
"""
import random
import os

from .augmentation_transforms import apply_policy


class AutoAugment(object):

    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        """

        #
        # ImageNet policies proposed in https://arxiv.org/abs/1805.09501
        #
        policies = [
            [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
            [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
            [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
            [('Posterize', 0.6, 7), ('Posterize', 0.6, 3)],
            [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
            [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
            [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
            [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
            [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
            [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
            [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
            [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
            [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
            [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
            [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
            [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
            [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
            [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
            [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
            [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
            [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
            [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
            [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
            [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
            [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        ]

        policy = random.choice(policies)

        img = apply_policy(policy, img)

        return img.convert('RGB')
