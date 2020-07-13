import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from .base_dataset import BaseDataset
from torch.utils.data import DataLoader, Dataset


class DLDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.train_sampler, self.val_sampler = None, None
        self.subset_sampler_used = False
        self.train_indices, self.val_indices = None, None

    def create_train_val_split(self, dataset: Dataset, train_val_split=0.2, shuffle=True):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_split = int(np.floor(train_val_split * dataset_size))

        if shuffle:
            np.random.seed(1)
            np.random.shuffle(indices)

        self.val_indices, self.train_indices = indices[:test_split], indices[test_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.subset_sampler_used = True

    def get_train_samples_num(self):
        raise NotImplementedError()

    def get_train_val_indices(self):
        return self.train_indices, self.val_indices

    def get_loader_labels(self, loader: DataLoader):
        labels = list()
        for i, data in enumerate(loader):
            if len(data) != 2:
                raise ValueError('No labels found!')
            labels.extend(list(data[1]))
        return np.asarray(labels)

    def get_labels(self, mode='val'):
        if mode == 'val':
            loader = DataLoader(dataset=self.train_dataset, batch_size=32, shuffle=False,
                                sampler=None,
                                num_workers=4)
            return self.get_loader_labels(loader)[self.val_indices]
        elif mode == 'train':
            loader = DataLoader(dataset=self.train_dataset, batch_size=32,
                                sampler=None, num_workers=4, shuffle=False)
            return self.get_loader_labels(loader)[self.train_indices]
        else:
            loader = DataLoader(dataset=self.test_dataset, batch_size=32, shuffle=False,
                                num_workers=4)
            return self.get_loader_labels(loader)
