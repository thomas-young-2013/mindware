from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition

from torchvision.transforms import transforms


def parse_bool(input):
    if input == 'True':
        return True
    else:
        return False


def get_aug_hyperparameter_space():
    cs = ConfigurationSpace()
    aug = CategoricalHyperparameter('aug', choices=['True', 'False'], default_value='True')
    auto_aug = CategoricalHyperparameter('auto_aug', choices=['True', 'False'], default_value='False')
    random_flip = CategoricalHyperparameter('random_flip', choices=['True', 'False'], default_value='True')
    affine = CategoricalHyperparameter('affine', choices=['True', 'False'], default_value='True')
    jitter = CategoricalHyperparameter('jitter', choices=['True', 'False'], default_value='True')
    brightness = CategoricalHyperparameter('brightness', choices=[0.2], default_value=0.2)
    saturation = CategoricalHyperparameter('saturation', choices=[0.2], default_value=0.2)
    hue = CategoricalHyperparameter('hue', choices=[0.15], default_value=0.15)
    degree = CategoricalHyperparameter('degree', choices=[10, 20, 30], default_value=10)
    shear = CategoricalHyperparameter('shear', choices=[0.05, 0.1, 0.2], default_value=0.1)

    cs.add_hyperparameters([aug, random_flip, auto_aug, affine, jitter, brightness, saturation, hue, degree, shear])

    auto_aug_on_aug = EqualsCondition(auto_aug, aug, 'True')
    random_flip_on_auto_aug = EqualsCondition(random_flip, auto_aug, 'False')
    affine_on_auto_aug = EqualsCondition(affine, auto_aug, 'False')
    jitter_on_auto_aug = EqualsCondition(jitter, auto_aug, 'False')
    brightness_on_jitter = EqualsCondition(brightness, jitter, 'True')
    saturation_on_jitter = EqualsCondition(saturation, jitter, 'True')
    hue_on_jitter = EqualsCondition(hue, jitter, 'True')
    degree_on_affine = EqualsCondition(degree, affine, 'True')
    shear_on_affine = EqualsCondition(shear, affine, 'True')

    cs.add_conditions([auto_aug_on_aug, random_flip_on_auto_aug, affine_on_auto_aug,
                       jitter_on_auto_aug, brightness_on_jitter, saturation_on_jitter,
                       hue_on_jitter, degree_on_affine, shear_on_affine])

    return cs


def get_transforms(config, image_size=256):
    config = config.get_dictionary()
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    if parse_bool(config['aug']):
        if parse_bool(config['auto_aug']):
            from .transforms import AutoAugment
            data_transforms = {
                'train': transforms.Compose([
                    AutoAugment(),
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'val': val_transforms,
            }
        else:
            transform_list = []
            if parse_bool(config['jitter']):
                transform_list.append(transforms.ColorJitter(brightness=config['brightness'],
                                                             saturation=config['saturation'],
                                                             hue=config['hue']))
            if parse_bool(config['affine']):
                transform_list.append(transforms.RandomAffine(degrees=config['degree'],
                                                              shear=config['shear']))

            transform_list.append(transforms.RandomResizedCrop(image_size))
            transform_list.append(transforms.RandomCrop(image_size, padding=4))

            if parse_bool(config['random_flip']):
                transform_list.append(transforms.RandomHorizontalFlip())

            transform_list.append(transforms.ToTensor())

            data_transforms = {'train': transforms.Compose(transform_list), 'val': val_transforms}
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]),
            'val': val_transforms,
        }
    return data_transforms


def get_test_transforms(image_size=256):
    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    return test_transforms
