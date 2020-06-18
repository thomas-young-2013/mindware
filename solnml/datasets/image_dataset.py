import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from solnml.components.models.img_classification.nn_utils.dataset import get_folder_dataset


class BaseDataset(object):
    def __init__(self):
        self.train_sampler, self.val_sampler = None, None
        self.subset_sampler_used = False

    def create_train_val_split(self, dataset: Dataset, train_val_split=0.2, shuffle=False):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_split = int(np.floor(train_val_split * dataset_size))

        if shuffle:
            np.random.shuffle(indices)

        val_indices, train_indices = indices[:test_split], indices[test_split:]

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        self.subset_sampler_used = True


class ImageDataset(BaseDataset):
    def __init__(self, data_path: str,
                 data_transforms: dict,
                 grayscale: bool = False,
                 train_val_split: bool = False,
                 val_split_size: float = 0.2):
        super().__init__()

        self.data_path = data_path
        self.udf_transforms = data_transforms
        self.train_val_split = train_val_split
        self.val_split_size = val_split_size
        self.grayscale = grayscale
        self.test_dataset, self.val_dataset = None, None

        self.train_dataset = get_folder_dataset(os.path.join(data_path, 'train'),
                                                udf_transforms=data_transforms['train'],
                                                grayscale=grayscale)
        if not self.train_val_split:
            self.val_dataset = get_folder_dataset(os.path.join(data_path, 'val'),
                                                  udf_transforms=data_transforms['val'],
                                                  grayscale=grayscale)
        else:
            self.create_train_val_split(self.train_dataset, train_val_split=val_split_size, shuffle=True)

    def load_test_data(self, test_data_path: str = None):
        if test_data_path is None:
            test_data_path = self.data_path
        self.test_dataset = get_folder_dataset(os.path.join(test_data_path, 'test'),
                                               udf_transforms=self.udf_transforms['val'],
                                               grayscale=self.grayscale)
