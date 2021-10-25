import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from .base_dataset import BaseDataset
from torch.utils.data import DataLoader, Dataset


class SubsetSequentialampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


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
        self.val_sampler = SubsetSequentialampler(self.val_indices)
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
            if self.subset_sampler_used:
                loader = DataLoader(dataset=self.train_dataset, batch_size=32,
                                    sampler=self.val_sampler, num_workers=4)
                return self.get_loader_labels(loader)
            else:
                loader = DataLoader(dataset=self.val_dataset, batch_size=32, shuffle=False,
                                    sampler=None, num_workers=4)
                return self.get_loader_labels(loader)
        elif mode == 'train':
            if self.subset_sampler_used:
                loader = DataLoader(dataset=self.train_dataset, batch_size=32,
                                    sampler=self.train_sampler, num_workers=4)
                return self.get_loader_labels(loader)
            else:
                loader = DataLoader(dataset=self.train_dataset, batch_size=32, shuffle=False,
                                    sampler=None, num_workers=4)
                return self.get_loader_labels(loader)
        else:
            loader = DataLoader(dataset=self.test_dataset, batch_size=32, shuffle=False,
                                num_workers=4)
            return self.get_loader_labels(loader)
