from __future__ import print_function, division, absolute_import
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR

from solnml.components.models.base_model import BaseClassificationModel
from solnml.components.models.img_classification.nn_utils.dataset import ArrayDataset


class BaseImgClassificationNeuralNetwork(BaseClassificationModel):
    def __init__(self):
        super(BaseImgClassificationNeuralNetwork, self).__init__()
        self.model = None
        self.learning_rate = None
        self.beta1 = None
        self.batch_size = None
        self.epoch_num = None
        self.lr_decay = None
        self.step_decay = None
        self.device = None

    def fit(self, X, y):
        assert self.model is not None
        params = self.model.parameters()
        loader = DataLoader(dataset=ArrayDataset(X, y), batch_size=self.batch_size, shuffle=True)
        optimizer = Adam(params=params, lr=self.learning_rate, betas=(self.beta1, 0.999))
        scheduler = StepLR(optimizer, step_size=self.step_decay, gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(self.epoch_num):
            for i, data in enumerate(loader):
                batch_x, batch_y = data['x'], data['y']
                logits = self.model(batch_x.float().to(self.device))
                optimizer.zero_grad()
                loss = loss_func(logits, batch_y.to(self.device))
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict_proba(self, X):
        if not self.model:
            raise ValueError("Model not fitted!")
        X = torch.Tensor(X)
        prediction = self.model(X)
        return prediction.detach().numpy()

    def predict(self, X):
        if not self.model:
            raise ValueError("Model not fitted!")
        X = torch.Tensor(X)
        prediction = self.model(X)
        return np.argmax(prediction.detach().numpy(), axis=-1)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        raise NotImplementedError()


# TODO: Characterize
class BaseTextClassificationNeuralNetwork(BaseClassificationModel):
    def __init__(self):
        super(BaseTextClassificationNeuralNetwork, self).__init__()
        self.model = None
        self.learning_rate = None
        self.beta1 = None
        self.batch_size = None
        self.epoch_num = None
        self.lr_decay = None
        self.step_decay = None
        self.device = None

    def fit(self, X, y):
        assert self.model is not None
        params = self.model.parameters()
        loader = DataLoader(dataset=ArrayDataset(X, y), batch_size=self.batch_size, shuffle=True)
        optimizer = Adam(params=params, lr=self.learning_rate, betas=(self.beta1, 0.999))
        scheduler = StepLR(optimizer, step_size=self.step_decay, gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(self.epoch_num):
            for i, data in enumerate(loader):
                batch_x, batch_y = data['x'], data['y']
                logits = self.model(batch_x.float().to(self.device))
                optimizer.zero_grad()
                loss = loss_func(logits, batch_y.to(self.device))
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict_proba(self, X):
        if not self.model:
            raise ValueError("Model not fitted!")
        X = torch.Tensor(X)
        prediction = self.model(X)
        return prediction.detach().numpy()

    def predict(self, X):
        if not self.model:
            raise ValueError("Model not fitted!")
        X = torch.Tensor(X)
        prediction = self.model(X)
        return np.argmax(prediction.detach().numpy(), axis=-1)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        raise NotImplementedError()
