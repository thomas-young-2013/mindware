import os
import sys
from torchvision import transforms
sys.path.append(os.getcwd())

from mindware.datasets.image_dataset import ImageDataset

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(560),
        transforms.RandomCrop(331),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(560),
        transforms.CenterCrop(331),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/img_datasets/hymenoptera_data'
image_data = ImageDataset(data_path=data_dir,
                          data_transforms=data_transforms)

print(image_data.train_dataset.classes)
print(image_data.val_dataset.classes)
image_data.load_test_data()
print(image_data.test_dataset.classes)

image_data = ImageDataset(data_path=data_dir,
                          data_transforms=data_transforms,
                          train_val_split=True,
                          val_split_size=0.3)

print(image_data.train_dataset.classes)
print(image_data.train_sampler, image_data.val_sampler)
