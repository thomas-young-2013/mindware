import os
import csv
import torch
from torch.utils.data import Dataset

from .base_dl_dataset import DLDataset


class TextBertDataset(Dataset):
    def __init__(self, csv_path,
                 padding_size=512,
                 config_path=None):
        """
        :param data: csv path, each line is (class_id, text)
        :param label: label name list
        """
        from transformers import BertTokenizer

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'bert-base-uncased')
        self.config_path = config_path
        self.path = csv_path
        self.padding_size = padding_size
        self._data = list()
        self.classes = set()
        for line in csv.reader(open(self.path, 'r')):
            self._data.append(line)
            self.classes.add(line[0])
        self._tokenizer = BertTokenizer.from_pretrained(config_path)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        sample = self._tokenizer.encode(self._data[item][1])
        return [torch.Tensor(self.padding(sample)), int(self._data[item][0])]

    def padding(self, sample):
        sample = sample + [0] * (self.padding_size - len(sample))
        return sample


class TextDataset(DLDataset):
    def __init__(self, data_path,
                 padding_size=512,
                 config_path=None,
                 train_val_split: bool = True,
                 val_split_size: float = 0.2):
        super().__init__()
        self.train_val_split = train_val_split
        self.val_split_size = val_split_size
        self.data_path = data_path

        self.padding_size = padding_size
        self.config_path = config_path

    def load_data(self):
        self.train_dataset = TextBertDataset(self.data_path, self.padding_size, self.config_path)
        self.classes = self.train_dataset.classes
        if self.train_val_split:
            self.create_train_val_split(self.train_dataset, train_val_split=self.val_split_size, shuffle=True)

    def load_test_data(self, test_data_path):
        self.test_dataset = TextBertDataset(test_data_path, self.padding_size, self.config_path)
