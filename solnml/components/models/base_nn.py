from __future__ import print_function, division, absolute_import
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from solnml.datasets.base_dl_dataset import DLDataset
from solnml.components.utils.dl_util import EarlyStop


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

    def set_empty_model(self, dataset):
        raise NotImplementedError


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

    def fit(self, dataset: DLDataset or Dataset):
        from sklearn.metrics import accuracy_score

        assert self.model is not None
        params = self.model.parameters()
        val_loader = None
        if isinstance(dataset, Dataset):
            train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            if hasattr(dataset, 'val_dataset'):
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=4)
                val_loader = DataLoader(dataset=dataset.val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=4)
            else:
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                          sampler=dataset.train_sampler, num_workers=4)
                val_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                        sampler=dataset.val_sampler, num_workers=4)

        if self.optimizer == 'SGD':
            optimizer = SGD(params=params, lr=self.sgd_learning_rate, momentum=self.sgd_momentum)
        elif self.optimizer == 'Adam':
            optimizer = Adam(params=params, lr=self.adam_learning_rate, betas=(self.beta1, 0.999))

        scheduler = StepLR(optimizer, step_size=self.step_decay, gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        self.model.train()

        early_stop = EarlyStop(patience=15, mode='min')

        for epoch in range(self.epoch_num):
            epoch_avg_loss = 0
            epoch_avg_acc = 0
            val_avg_loss = 0
            val_avg_acc = 0
            num_train_samples = 0
            num_val_samples = 0
            for i, data in enumerate(train_loader):
                batch_x, batch_y = data[0], data[1]
                logits = self.model(batch_x.float().to(self.device))
                optimizer.zero_grad()
                loss = loss_func(logits, batch_y.to(self.device))
                num_train_samples += len(batch_x)
                loss.backward()
                epoch_avg_loss += loss.to('cpu').detach() * len(batch_x)
                optimizer.step()

                prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                epoch_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

            epoch_avg_loss /= num_train_samples
            epoch_avg_acc /= num_train_samples
            # TODO: logger
            print('Epoch %d: Train loss %.4f, train acc %.4f' % (epoch, epoch_avg_loss, epoch_avg_acc))

            if val_loader is not None:
                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        batch_x, batch_y = data[0], data[1]
                        logits = self.model(batch_x.float().to(self.device))
                        val_loss = loss_func(logits, batch_y.to(self.device))
                        num_val_samples += len(batch_x)
                        val_avg_loss += val_loss.to('cpu').detach() * len(batch_x)

                        prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                        val_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

                    val_avg_loss /= num_val_samples
                    val_avg_acc /= num_val_samples
                    print('Epoch %d: Val loss %.4f, val acc %.4f' % (epoch, val_avg_loss, val_avg_acc))

                    # Early stop
                    early_stop.update(val_avg_loss)
                    if early_stop.if_early_stop:
                        print("Early stop!")
                        break

            scheduler.step()
        return self

    def predict_proba(self, dataset: Dataset, sampler=None, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
        self.model.to(self.device)
        self.model.eval()

        prediction = None
        with torch.no_grad():
            for i, data in enumerate(loader):
                batch_x, batch_y = data[0], data[1]
                logits = self.model(batch_x.float().to(self.device))
                pred = nn.functional.softmax(logits, dim=-1)
                if prediction is None:
                    prediction = pred.to('cpu').detach().numpy()
                else:
                    prediction = np.concatenate((prediction, pred.to('cpu').detach().numpy()), 0)

        return prediction

    def predict(self, dataset: Dataset, sampler=None, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
        self.model.to(self.device)
        self.model.eval()

        prediction = None
        with torch.no_grad():
            for i, data in enumerate(loader):
                batch_x, batch_y = data[0], data[1]
                logits = self.model(batch_x.float().to(self.device))
                if prediction is None:
                    prediction = logits.to('cpu').detach().numpy()
                else:
                    prediction = np.concatenate((prediction, logits.to('cpu').detach().numpy()), 0)
        return np.argmax(prediction, axis=-1)

    def score(self, dataset, metric, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        if isinstance(dataset, Dataset):
            loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
        else:
            if hasattr(dataset, 'val_dataset'):
                loader = DataLoader(dataset=dataset.val_dataset, batch_size=batch_size, num_workers=4)
            else:
                loader = DataLoader(dataset=dataset.train_dataset, batch_size=batch_size,
                                    sampler=dataset.val_sampler, num_workers=4)

        self.model.to(self.device)
        self.model.eval()
        total_len = 0
        score = 0
        with torch.no_grad():
            for i, data in enumerate(loader):
                batch_x, batch_y = data[0], data[1]
                logits = self.model(batch_x.float().to(self.device)).to('cpu')
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
        from sklearn.metrics import accuracy_score

        assert self.model is not None
        params = self.model.parameters()
        val_loader = None
        if isinstance(dataset, Dataset):
            train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            if hasattr(dataset, 'val_dataset'):
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=4)
                val_loader = DataLoader(dataset=dataset.val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=4)
            else:
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                          sampler=dataset.train_sampler, num_workers=4)
                val_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                        sampler=dataset.val_sampler, num_workers=4)

        if self.optimizer == 'SGD':
            optimizer = SGD(params=params, lr=self.sgd_learning_rate, momentum=self.sgd_momentum)
        elif self.optimizer == 'Adam':
            optimizer = Adam(params=params, lr=self.adam_learning_rate, betas=(self.beta1, 0.999))

        scheduler = StepLR(optimizer, step_size=self.step_decay, gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        self.model.train()

        early_stop = EarlyStop(patience=15, mode='min')

        for epoch in range(self.epoch_num):
            epoch_avg_loss = 0
            epoch_avg_acc = 0
            val_avg_loss = 0
            val_avg_acc = 0
            num_train_samples = 0
            num_val_samples = 0
            for i, data in enumerate(train_loader):
                batch_x, batch_y = data[0], data[1]
                masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
                logits = self.model(batch_x.long().to(self.device), masks.to(self.device))
                optimizer.zero_grad()
                loss = loss_func(logits, batch_y.to(self.device))
                num_train_samples += len(batch_x)
                loss.backward()
                epoch_avg_loss += loss.to('cpu').detach() * len(batch_x)
                optimizer.step()

                prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                epoch_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

            epoch_avg_loss /= num_train_samples
            epoch_avg_acc /= num_train_samples
            # TODO: logger
            print('Epoch %d: Train loss %.4f, train acc %.4f' % (epoch, epoch_avg_loss, epoch_avg_acc))

            if val_loader is not None:
                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        batch_x, batch_y = data[0], data[1]
                        masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
                        logits = self.model(batch_x.long().to(self.device), masks.to(self.device))
                        val_loss = loss_func(logits, batch_y.to(self.device))
                        num_val_samples += len(batch_x)
                        val_avg_loss += val_loss.to('cpu').detach() * len(batch_x)

                        prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                        val_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

                    val_avg_loss /= num_val_samples
                    val_avg_acc /= num_val_samples
                    print('Epoch %d: Val loss %.4f, val acc %.4f' % (epoch, val_avg_loss, val_avg_acc))

                    # Early stop
                    early_stop.update(val_avg_loss)
                    if early_stop.if_early_stop:
                        print("Early stop!")
                        break

            scheduler.step()
        return self

    def predict_proba(self, dataset: Dataset, sampler=None, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
        self.model.to(self.device)
        self.model.eval()

        prediction = None
        with torch.no_grad():
            for i, data in enumerate(loader):
                batch_x, batch_y = data[0], data[1]
                masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
                logits = self.model(batch_x.long().to(self.device), masks.to(self.device))
                pred = nn.functional.softmax(logits, dim=-1)
                if prediction is None:
                    prediction = pred.to('cpu').detach().numpy()
                else:
                    prediction = np.concatenate((prediction, pred.to('cpu').detach().numpy()), 0)
        return prediction

    def predict(self, dataset: Dataset, sampler=None, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
        self.model.to(self.device)
        self.model.eval()

        prediction = None
        with torch.no_grad():
            for i, data in enumerate(loader):
                batch_x, batch_y = data[0], data[1]
                masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
                logits = self.model(batch_x.long().to(self.device), masks.to(self.device))
                if prediction is None:
                    prediction = logits.to('cpu').detach().numpy()
                else:
                    prediction = np.concatenate((prediction, logits.to('cpu').detach().numpy()), 0)
        return np.argmax(prediction, axis=-1)

    def score(self, dataset, metric, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        if isinstance(dataset, Dataset):
            loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
        else:
            if hasattr(dataset, 'val_dataset'):
                loader = DataLoader(dataset=dataset.val_dataset, batch_size=batch_size, num_workers=4)
            else:
                loader = DataLoader(dataset=dataset.train_dataset, batch_size=batch_size,
                                    sampler=dataset.val_sampler, num_workers=4)
        self.model.eval()
        total_len = 0
        score = 0
        with torch.no_grad():
            for i, data in enumerate(loader):
                batch_x, batch_y = data[0], data[1]
                masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
                logits = self.model(batch_x.long().to(self.device), masks.to(self.device)).to('cpu')
                prediction = np.argmax(logits.detach().numpy(), axis=-1)
                score += metric(prediction, batch_y.detach().numpy()) * len(prediction)
                total_len += len(prediction)
        score /= total_len
        return score

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

    def fit(self, dataset: DLDataset or Dataset):
        assert self.model is not None
        params = self.model.parameters()

        val_loader = None
        if isinstance(dataset, Dataset):
            train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            if hasattr(dataset, 'val_dataset'):
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=4, collate_fn=dataset.train_dataset.collate_fn)
                val_loader = DataLoader(dataset=dataset.val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=4, collate_fn=dataset.val_dataset.collate_fn)
            else:
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                          sampler=dataset.train_sampler, num_workers=4,
                                          collate_fn=dataset.train_dataset.collate_fn)
                val_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                        sampler=dataset.val_sampler, num_workers=4,
                                        collate_fn=dataset.train_dataset.collate_fn)

        if self.optimizer == 'SGD':
            optimizer = SGD(params=params, lr=self.sgd_learning_rate, momentum=self.sgd_momentum)
        elif self.optimizer == 'Adam':
            optimizer = Adam(params=params, lr=self.adam_learning_rate, betas=(self.beta1, 0.999))

        scheduler = StepLR(optimizer, step_size=self.step_decay, gamma=self.lr_decay)
        self.model.train()

        early_stop = EarlyStop(patience=15, mode='min')

        for epoch in range(self.epoch_num):
            epoch_avg_loss = 0
            val_avg_loss = 0
            num_train_samples = 0
            num_val_samples = 0
            for i, (_, batch_x, batch_y) in enumerate(train_loader):
                loss, outputs = self.model(batch_x.float().to(self.device), batch_y.float().to(self.device))
                optimizer.zero_grad()
                epoch_avg_loss += loss.to('cpu').detach() * len(batch_x)
                num_train_samples += len(batch_x)
                loss.backward()
                optimizer.step()
            epoch_avg_loss /= num_train_samples
            print('Epoch %d: Train loss %.4f' % (epoch, epoch_avg_loss))
            scheduler.step()

            if val_loader is not None:
                with torch.no_grad():
                    for i, (_, batch_x, batch_y) in enumerate(val_loader):
                        loss, outputs = self.model(batch_x.float().to(self.device), batch_y.float().to(self.device))
                        val_avg_loss += loss.to('cpu').detach() * len(batch_x)
                        num_val_samples += len(batch_x)

                    val_avg_loss /= num_val_samples
                    print('Epoch %d: Val loss %.4f' % (epoch, val_avg_loss))

                # Early stop
                early_stop.update(val_avg_loss)
                if early_stop.if_early_stop:
                    print("Early stop!")
                    break

        return self

    def predict(self, dataset: Dataset, sampler=None, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=4, collate_fn=dataset.collate_fn)
        self.model.to(self.device)
        self.model.eval()

        prediction = None
        with torch.no_grad():
            for i, data in enumerate(loader):
                batch_x, batch_y = data[0], data[1]
                logits = self.model(batch_x.float().to(self.device))
                if prediction is None:
                    prediction = logits.to('cpu').detach().numpy()
                else:
                    prediction = np.concatenate((prediction, logits.to('cpu').detach().numpy()), 0)
        return np.argmax(prediction, axis=-1)

    # TODO: UDF metric
    def score(self, dataset, metric, batch_size=None):
        raise NotImplementedError

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        raise NotImplementedError()
