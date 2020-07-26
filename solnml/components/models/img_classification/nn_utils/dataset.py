from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ArrayDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.classes = set(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return [self.x[item], self.y[item]]


def get_array_dataset(X, y):
    return ArrayDataset(X, y)


def get_folder_dataset(folder_path, udf_transforms=None, grayscale=False):
    return datasets.ImageFolder(folder_path, transform=udf_transforms)
