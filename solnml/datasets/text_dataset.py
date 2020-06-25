from .base_dataset import BaseDataset
from solnml.components.models.text_classification.nn_utils.dataset import TextBertDataset


class TextDataset(BaseDataset):
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
        if self.train_val_split:
            self.create_train_val_split(self.train_dataset, train_val_split=self.val_split_size, shuffle=True)

    def load_test_data(self, test_data_path):
        self.train_dataset = TextBertDataset(test_data_path, self.padding_size, self.config_path)
