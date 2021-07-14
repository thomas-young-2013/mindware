import numpy as np


class EarlyStop:
    def __init__(self, patience=20, mode='min'):
        self.patience = patience
        assert mode in ['max', 'min']
        self.mode = mode
        if mode == 'min':
            self.cur_value = np.inf
        else:
            self.cur_value = -np.inf
        self.cur_patience = 0
        self.if_early_stop = False

    def update(self, val_value):
        if_update = self._check_value(val_value)
        if if_update:
            self.cur_value = val_value
            self.cur_patience = 0
        else:
            self.cur_patience += 1
            if self.cur_patience > self.patience:
                self.if_early_stop = True

    def _check_value(self, val_value):
        """
        :param val_value:
        :return: True if val_value is better than the current value
        """
        if self.mode == 'min':
            return val_value < self.cur_value
        else:
            return val_value > self.cur_value
