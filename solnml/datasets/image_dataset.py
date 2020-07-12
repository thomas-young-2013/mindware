import os
from .base_dl_dataset import DLDataset
from torch.utils.data import DataLoader, Dataset
from solnml.components.models.img_classification.nn_utils.dataset import get_folder_dataset


class ImageDataset(DLDataset):
    def __init__(self, data_path: str,
                 data_transforms: dict = None,
                 grayscale: bool = False,
                 train_val_split: bool = False,
                 val_split_size: float = 0.2):
        super().__init__()
        self.train_val_split = train_val_split
        self.val_split_size = val_split_size
        self.data_path = data_path

        self.udf_transforms = data_transforms
        self.grayscale = grayscale

        default_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'))
        self.classes = default_dataset.classes

    def load_data(self, train_transforms, val_transforms):
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

    def get_num_train_samples(self):
        self.load_data(None, None)
        if self.subset_sampler_used:
            return len(list(self.train_sampler))
        else:
            return len(self.train_dataset)

