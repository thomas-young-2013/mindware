import os
import csv
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TextBertDataset(Dataset):
    def __init__(self, csv_path,
                 padding_size=512,
                 config_path=None):
        """
        :param data: csv path, each line is (class_id, text)
        :param label: label name list
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'bert-base-uncased')
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
