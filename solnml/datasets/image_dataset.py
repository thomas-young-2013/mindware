import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .base_dl_dataset import DLDataset
from solnml.components.models.img_classification.nn_utils.dataset import get_folder_dataset


class ImageDataset(DLDataset):
    def __init__(self, data_path: str,
                 data_transforms: dict = None,
                 grayscale: bool = False,
                 train_val_split: bool = False,
                 image_size=32,
                 val_split_size: float = 0.2):
        super().__init__()
        self.train_val_split = train_val_split
        self.val_split_size = val_split_size
        self.data_path = data_path

        self.udf_transforms = data_transforms
        self.grayscale = grayscale
        self.image_size = image_size

        default_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'))
        self.classes = default_dataset.classes

    def load_data(self, train_transforms, val_transforms):
        # self.means, self.var = self.get_mean_and_var()
        self.train_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'),
                                                udf_transforms=train_transforms,
                                                grayscale=self.grayscale)
        if not self.train_val_split:
            self.val_dataset = get_folder_dataset(os.path.join(self.data_path, 'val'),
                                                  udf_transforms=val_transforms,
                                                  grayscale=self.grayscale)
        else:
            self.train_for_val_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'),
                                                            udf_transforms=val_transforms,
                                                            grayscale=self.grayscale)
            self.create_train_val_split(self.train_dataset, train_val_split=self.val_split_size, shuffle=True)

    def load_test_data(self, transforms):
        self.test_dataset = get_folder_dataset(os.path.join(self.test_data_path, 'test'),
                                               udf_transforms=transforms,
                                               grayscale=self.grayscale)
        self.test_dataset.classes = self.classes

    def get_train_samples_num(self):
        if self.train_dataset is None:
            _train_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'),
                                                udf_transforms=None,
                                                grayscale=self.grayscale)
            _train_size = len(_train_dataset)
        else:
            _train_size = len(self.train_dataset)
        if self.subset_sampler_used:
            return _train_size * (1 - self.val_split_size)
        else:
            return _train_size

    def get_mean_and_var(self):
        basic_transforms = transforms.Compose([
            transforms.ToTensor()])
        _train_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'),
                                            udf_transforms=basic_transforms)

        dataloader = torch.utils.data.DataLoader(_train_dataset, batch_size=1, shuffle=True, num_workers=2)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            for i in range(3):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()
        mean.div_(len(_train_dataset))
        std.div_(len(_train_dataset))
        mean = mean.numpy()
        std = std.numpy()
        return mean, std
