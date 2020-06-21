from __future__ import print_function, division, absolute_import
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR


class BaseNeuralNetwork:
    @staticmethod
    def get_properties():
        """
        Get the properties of the underlying algorithm.
        :return: algorithm_properties : dict, optional (default=None)
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space():
        """
        Get the configuration space of this classification algorithm.
        :return: Configspace.configuration_space.ConfigurationSpace
            The configuration space of this classification algorithm.
        """
        raise NotImplementedError()

    def fit(self, dataset):
        """
        The fit function calls the fit function of the underlying model and returns `self`.
        :param dataset: torch.utils.data.Dataset
        :return: self, an instance of self.
        """
        raise NotImplementedError()

    def set_hyperparameters(self, params, init_params=None):
        """
        The function set the class members according to params
        :param params: dictionary, parameters
        :param init_params: dictionary
        :return:
        """
        for param, value in params.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' % (param, str(self)))
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError('Cannot set init param %s for %s because '
                                     'the init param does not exist.' %
                                     (param, str(self)))
                setattr(self, param, value)
        return self


class BaseImgClassificationNeuralNetwork(BaseNeuralNetwork):
    def __init__(self):
        super(BaseImgClassificationNeuralNetwork, self).__init__()
        self.model = None
        self.optimizer = None
        self.sgd_learning_rate = None
        self.sgd_momentum = None
        self.adam_learning_rate = None
        self.beta1 = None
        self.batch_size = None
        self.epoch_num = None
        self.lr_decay = None
        self.step_decay = None
        self.device = None

    def fit(self, dataset):
        assert self.model is not None
        params = self.model.parameters()
        if hasattr(dataset, 'val_dataset'):
            loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                sampler=dataset.train_sampler, shuffle=True, num_workers=4)
        if self.optimizer == 'SGD':
            optimizer = SGD(params=params, lr=self.sgd_learning_rate, momentum=self.sgd_momentum)
        elif self.optimizer == 'Adam':
            optimizer = Adam(params=params, lr=self.adam_learning_rate, betas=(self.beta1, 0.999))

        scheduler = StepLR(optimizer, step_size=self.step_decay, gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(self.epoch_num):
            for i, data in enumerate(loader):
                batch_x, batch_y = data[0], data[1]
                logits = self.model(batch_x.float().to(self.device))
                optimizer.zero_grad()
                loss = loss_func(logits, batch_y.to(self.device))
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict_proba(self, dataset, set='test', batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        if set == 'val':
            if hasattr(dataset, 'val_dataset'):
                loader = DataLoader(dataset=dataset.val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            else:
                loader = DataLoader(dataset=dataset.train_dataset, batch_size=batch_size,
                                    sampler=dataset.val_sampler, shuffle=True, num_workers=4)
        elif set == 'test':
            loader = DataLoader(dataset=dataset.test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.model.eval()
        prediction = torch.Tensor()
        for i, data in enumerate(loader):
            batch_x, batch_y = data[0], data[1]
            logits = self.model(batch_x.float().to(self.device))
            prediction = torch.cat((prediction, logits), 0)
        return prediction.detach().numpy()

    def predict(self, dataset, set='test', batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        if set == 'val':
            if hasattr(dataset, 'val_dataset'):
                loader = DataLoader(dataset=dataset.val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            else:
                loader = DataLoader(dataset=dataset.train_dataset, batch_size=batch_size,
                                    sampler=dataset.val_sampler, shuffle=True, num_workers=4)
        elif set == 'test':
            loader = DataLoader(dataset=dataset.test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.model.eval()
        prediction = torch.Tensor()
        for i, data in enumerate(loader):
            batch_x, batch_y = data[0], data[1]
            logits = self.model(batch_x.float().to(self.device))
            prediction = torch.cat((prediction, logits), 0)
        return np.argmax(prediction.detach().numpy(), axis=-1)

    def score(self, dataset, metric, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        if hasattr(dataset, 'val_dataset'):
            loader = DataLoader(dataset=dataset.val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            loader = DataLoader(dataset=dataset.train_dataset, batch_size=batch_size,
                                sampler=dataset.val_sampler, shuffle=True, num_workers=4)
        self.model.eval()
        total_len = 0
        score = 0
        for i, data in enumerate(loader):
            batch_x, batch_y = data[0], data[1]
            logits = self.model(batch_x.float().to(self.device))
            prediction = np.argmax(logits.detach().numpy(), axis=-1)
            score += metric(prediction, batch_y.detach().numpy()) * len(prediction)
            total_len += len(prediction)
        score /= total_len
        return score

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        raise NotImplementedError()


class BaseTextClassificationNeuralNetwork(BaseNeuralNetwork):
    def __init__(self):
        super(BaseTextClassificationNeuralNetwork, self).__init__()
        self.model = None
        self.optimizer = None
        self.sgd_learning_rate = None
        self.sgd_momentum = None
        self.adam_learning_rate = None
        self.beta1 = None
        self.batch_size = None
        self.epoch_num = None
        self.lr_decay = None
        self.step_decay = None
        self.device = None

    def fit(self, dataset):
        assert self.model is not None
        params = self.model.parameters()
        if hasattr(dataset, 'val_dataset'):
            loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                sampler=dataset.train_sampler, shuffle=True, num_workers=4)
        if self.optimizer == 'SGD':
            optimizer = SGD(params=params, lr=self.sgd_learning_rate, momentum=self.sgd_momentum)
        elif self.optimizer == 'Adam':
            optimizer = Adam(params=params, lr=self.adam_learning_rate, betas=(self.beta1, 0.999))

        scheduler = StepLR(optimizer, step_size=self.step_decay, gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(self.epoch_num):
            for i, data in enumerate(loader):
                batch_x, batch_y = data[0], data[1]
                masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
                logits = self.model(batch_x.long().to(self.device), masks)
                optimizer.zero_grad()
                loss = loss_func(logits, batch_y.to(self.device))
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict_proba(self, dataset, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.model.eval()
        prediction = torch.Tensor()
        for i, data in enumerate(loader):
            batch_x, batch_y = data[0], data[1]
            masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
            logits = self.model(batch_x.long().to(self.device), masks)
            prediction = torch.cat((prediction, logits), 0)
        return prediction.detach().numpy()

    def predict(self, dataset, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.model.eval()
        prediction = torch.Tensor()
        for i, data in enumerate(loader):
            batch_x, batch_y = data[0], data[1]
            masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
            logits = self.model(batch_x.long().to(self.device), masks)
            prediction = torch.cat((prediction, logits), 0)
        return np.argmax(prediction.detach().numpy(), axis=-1)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        raise NotImplementedError()


class BaseODClassificationNeuralNetwork(BaseNeuralNetwork):
    def __init__(self):
        super(BaseODClassificationNeuralNetwork, self).__init__()
        self.model = None
        self.optimizer = None
        self.sgd_learning_rate = None
        self.sgd_momentum = None
        self.adam_learning_rate = None
        self.beta1 = None
        self.batch_size = None
        self.epoch_num = None
        self.lr_decay = None
        self.step_decay = None
        self.device = None

    def fit(self, dataset):
        # TODO: Define OD process
        assert self.model is not None
        params = self.model.parameters()
        loader = DataLoader(dataset=ArrayDataset(X, targets), batch_size=self.batch_size, shuffle=True)
        if self.optimizer == 'SGD':
            optimizer = SGD(params=params, lr=self.sgd_learning_rate, momentum=self.sgd_momentum)
        elif self.optimizer == 'Adam':
            optimizer = Adam(params=params, lr=self.adam_learning_rate, betas=(self.beta1, 0.999))

        scheduler = StepLR(optimizer, step_size=self.step_decay, gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        self.model.train()
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

    def predict(self, X):
        if not self.model:
            raise ValueError("Model not fitted!")
        self.model.eval()
        X = torch.Tensor(X)
        return self.model(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        raise NotImplementedError()
