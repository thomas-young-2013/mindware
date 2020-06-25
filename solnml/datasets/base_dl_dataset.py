import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from .base_dataset import BaseDataset


class DLDataset(BaseDataset):
    def __init__(self):
        super().__init__()
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
