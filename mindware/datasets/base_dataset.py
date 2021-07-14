class BaseDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.test_data_path = None

    def load_data(self):
        raise NotImplementedError()

    def load_test_data(self):
        raise NotImplementedError()

    def set_test_path(self, test_data_path):
        self.test_data_path = test_data_path
