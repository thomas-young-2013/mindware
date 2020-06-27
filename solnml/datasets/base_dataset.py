class BaseDataset(object):
    def __init__(self):
        self.train_dataset = None

    def load_data(self):
        raise NotImplementedError()

    def load_test_data(self, data_path):
        raise NotImplementedError()
