from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return {'x': self.x[item], 'y': self.y[item]}
