from __future__ import print_function, division, absolute_import
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter

from mindware.datasets.base_dl_dataset import DLDataset
from mindware.components.utils.dl_util import EarlyStop
from mindware.components.utils.configspace_utils import check_for_bool

NUM_WORKERS = 10


class BaseNeuralNetwork:
    def __init__(self):
        self.early_stop_flag = False

    @staticmethod
    def get_properties():
        """
        Get the properties of the underlying algorithm.
        :return: algorithm_properties : dict, optional (default=None)
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

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        cs = ConfigurationSpace()
        optimizer = CategoricalHyperparameter('optimizer', ['SGD'], default_value='SGD')
        sgd_learning_rate = CategoricalHyperparameter(
            "sgd_learning_rate", [1e-3, 3e-3, 7e-3, 1e-2, 3e-2, 7e-2, 1e-1],
            default_value=1e-1)
        sgd_momentum = UniformFloatHyperparameter(
            "sgd_momentum", lower=0.5, upper=0.99, default_value=0.9, log=False)
        nesterov = CategoricalHyperparameter('nesterov', ['True', 'False'], default_value='True')

        batch_size = CategoricalHyperparameter(
            "batch_size", [32, 64, 128], default_value=32)
        lr_decay = CategoricalHyperparameter("lr_decay", [1e-2, 5e-2, 1e-1, 2e-1], default_value=1e-1)
        weight_decay = CategoricalHyperparameter("weight_decay", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
                                                 default_value=1e-4)
        epoch_num = UnParametrizedHyperparameter("epoch_num", 150)
        cs.add_hyperparameters(
            [optimizer, sgd_learning_rate, sgd_momentum, batch_size, epoch_num,
             lr_decay, weight_decay, nesterov])
        # optimizer = CategoricalHyperparameter('optimizer', ['SGD', 'Adam'], default_value='SGD')
        # adam_learning_rate = UniformFloatHyperparameter(
        #     "adam_learning_rate", lower=1e-4, upper=1e-2, default_value=2e-3, log=True)
        # beta1 = UniformFloatHyperparameter(
        #     "beta1", lower=0.5, upper=0.999, default_value=0.9, log=False)
        # batch_size = CategoricalHyperparameter(
        #     "batch_size", [16, 32, 64, 128], default_value=32)
        # sgd_lr_depends_on_sgd = EqualsCondition(sgd_learning_rate, optimizer, "SGD")
        # adam_lr_depends_on_adam = EqualsCondition(adam_learning_rate, optimizer, "Adam")
        # beta_depends_on_adam = EqualsCondition(beta1, optimizer, "Adam")
        # sgd_momentum_depends_on_sgd = EqualsCondition(sgd_momentum, optimizer, "SGD")
        # nesterov_depends_on_sgd = EqualsCondition(nesterov, optimizer, 'SGD')
        # cs.add_conditions(
        #     [sgd_lr_depends_on_sgd, sgd_momentum_depends_on_sgd,
        #      nesterov_depends_on_sgd])
        return cs


class BaseImgClassificationNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, optimizer, batch_size, epoch_num, lr_decay, weight_decay,
                 sgd_learning_rate=None, sgd_momentum=None, nesterov=None,
                 adam_learning_rate=None, beta1=None, random_state=None,
                 grayscale=False, device='cpu', **kwargs):
        super(BaseImgClassificationNeuralNetwork, self).__init__()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epoch = epoch_num
        self.epoch_num = epoch_num
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_momentum = sgd_momentum
        self.nesterov = check_for_bool(nesterov)
        self.adam_learning_rate = adam_learning_rate
        self.beta1 = beta1
        self.random_state = random_state
        self.grayscale = grayscale
        self.model = None
        self.device = torch.device(device)
        self.time_limit = None
        self.load_path = None

        self.optimizer_ = None
        self.scheduler = None
        self.early_stop = None
        self.cur_epoch_num = 0

    def fit(self, dataset: DLDataset, mode='fit', **kwargs):
        from sklearn.metrics import accuracy_score

        assert self.model is not None

        params = self.model.parameters()
        val_loader = None
        if 'refit' in mode:
            train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=NUM_WORKERS)
            if mode == 'refit_test':
                val_loader = DataLoader(dataset=dataset.test_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=NUM_WORKERS)
        else:
            if not dataset.subset_sampler_used:
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=NUM_WORKERS)
                val_loader = DataLoader(dataset=dataset.val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=NUM_WORKERS)
            else:
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                          sampler=dataset.train_sampler, num_workers=NUM_WORKERS)
                val_loader = DataLoader(dataset=dataset.train_for_val_dataset, batch_size=self.batch_size,
                                        sampler=dataset.val_sampler, num_workers=NUM_WORKERS)

        if self.optimizer == 'SGD':
            optimizer = SGD(params=params, lr=self.sgd_learning_rate, momentum=self.sgd_momentum,
                            weight_decay=self.weight_decay, nesterov=self.nesterov)
        elif self.optimizer == 'Adam':
            optimizer = Adam(params=params, lr=self.adam_learning_rate, betas=(self.beta1, 0.999),
                             weight_decay=self.weight_decay)
        else:
            return ValueError("Optimizer %s not supported!" % self.optimizer)

        scheduler = MultiStepLR(optimizer, milestones=[int(self.max_epoch * 0.5), int(self.max_epoch * 0.75)],
                                gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        early_stop = EarlyStop(patience=100, mode='min')

        if self.load_path:
            checkpoint = torch.load(self.load_path)
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            self.cur_epoch_num = checkpoint['epoch_num']
            early_stop = checkpoint['early_stop']
            if early_stop.if_early_stop:
                print("Early stop!")
                self.optimizer_ = optimizer
                self.epoch_num = int(self.epoch_num) + int(self.cur_epoch_num)
                self.scheduler = scheduler
                self.early_stop = early_stop
                return self

        profile_iter = kwargs.get('profile_iter', None)
        profile_epoch = kwargs.get('profile_epoch', None)
        assert not (profile_iter and profile_epoch)

        if profile_epoch or profile_iter:  # Profile mode
            self.model.train()
            if profile_epoch:
                for epoch in range(int(profile_epoch)):
                    for i, data in enumerate(train_loader):
                        batch_x, batch_y = data[0], data[1]
                        logits = self.model(batch_x.float().to(self.device))
                        optimizer.zero_grad()
                        loss = loss_func(logits, batch_y.to(self.device))
                        loss.backward()
                        optimizer.step()
            else:
                num_iter = 0
                stop_flag = False
                for epoch in range(int(self.epoch_num)):
                    if stop_flag:
                        break
                    for i, data in enumerate(train_loader):
                        batch_x, batch_y = data[0], data[1]
                        logits = self.model(batch_x.float().to(self.device))
                        optimizer.zero_grad()
                        loss = loss_func(logits, batch_y.to(self.device))
                        loss.backward()
                        optimizer.step()
                        num_iter += 1
                        if num_iter > profile_iter:
                            stop_flag = True
                            break
            return self

        for epoch in range(int(self.cur_epoch_num), int(self.cur_epoch_num) + int(self.epoch_num)):
            self.model.train()
            # print('Current learning rate: %.5f' % optimizer.state_dict()['param_groups'][0]['lr'])
            epoch_avg_loss = 0
            epoch_avg_acc = 0
            val_avg_loss = 0
            val_avg_acc = 0
            num_train_samples = 0
            num_val_samples = 0
            for i, data in enumerate(train_loader):
                batch_x, batch_y = data[0], data[1]
                num_train_samples += len(batch_x)
                logits = self.model(batch_x.float().to(self.device))
                loss = loss_func(logits, batch_y.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_avg_loss += loss.to('cpu').detach() * len(batch_x)
                prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                epoch_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

            epoch_avg_loss /= num_train_samples
            epoch_avg_acc /= num_train_samples
            # TODO: logger
            print('Epoch %d: Train loss %.4f, train acc %.4f' % (epoch, epoch_avg_loss, epoch_avg_acc))

            if val_loader is not None:
                self.model.eval()
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
                    if 'refit' not in mode:
                        early_stop.update(val_avg_loss)
                        if early_stop.if_early_stop:
                            self.early_stop_flag = True
                            print("Early stop!")
                            break

            scheduler.step()

        self.optimizer_ = optimizer
        self.epoch_num = int(self.epoch_num) + int(self.cur_epoch_num)
        self.scheduler = scheduler
        self.early_stop = early_stop

        return self

    def predict_proba(self, dataset: Dataset, sampler=None, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS)
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
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS)
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
            loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
        else:
            if not dataset.subset_sampler_used:
                loader = DataLoader(dataset=dataset.val_dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
            else:
                loader = DataLoader(dataset=dataset.train_for_val_dataset, batch_size=batch_size,
                                    sampler=dataset.val_sampler, num_workers=NUM_WORKERS)

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


class BaseTextClassificationNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, optimizer, batch_size, epoch_num, lr_decay, step_decay, weight_decay,
                 sgd_learning_rate=None, sgd_momentum=None, nesterov=None, adam_learning_rate=None,
                 beta1=None, random_state=None, device='cpu',
                 config='./mindware/components/models/text_classification/nn_utils/bert-base-uncased'):
        super(BaseTextClassificationNeuralNetwork, self).__init__()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epoch = epoch_num
        self.epoch_num = epoch_num
        self.lr_decay = lr_decay
        self.step_decay = step_decay
        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_momentum = sgd_momentum
        self.adam_learning_rate = adam_learning_rate
        self.beta1 = beta1
        self.random_state = random_state
        self.model = None
        self.device = torch.device(device)
        self.time_limit = None
        self.config = config
        self.load_path = None

        self.optimizer_ = None
        self.scheduler = None
        self.early_stop = None
        self.cur_epoch_num = 0

    def fit(self, dataset, mode='fit', **kwargs):
        from sklearn.metrics import accuracy_score

        assert self.model is not None

        params = self.model.parameters()
        val_loader = None
        if 'refit' in mode:
            train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=NUM_WORKERS)
            if mode == 'refit_test':
                val_loader = DataLoader(dataset=dataset.test_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=NUM_WORKERS)
        else:
            if not dataset.subset_sampler_used:
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=NUM_WORKERS)
                val_loader = DataLoader(dataset=dataset.val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=NUM_WORKERS)
            else:
                train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
                                          sampler=dataset.train_sampler, num_workers=NUM_WORKERS)
                val_loader = DataLoader(dataset=dataset.train_for_val_dataset, batch_size=self.batch_size,
                                        sampler=dataset.val_sampler, num_workers=NUM_WORKERS)

        if self.optimizer == 'SGD':
            optimizer = SGD(params=params, lr=self.sgd_learning_rate, momentum=self.sgd_momentum)
        elif self.optimizer == 'Adam':
            optimizer = Adam(params=params, lr=self.adam_learning_rate, betas=(self.beta1, 0.999))
        else:
            return ValueError("Optimizer %s not supported!" % self.optimizer)

        scheduler = MultiStepLR(optimizer, milestones=[int(self.max_epoch * 0.5), int(self.max_epoch * 0.75)],
                                gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        early_stop = EarlyStop(patience=5, mode='min')

        if self.load_path:
            checkpoint = torch.load(self.load_path)
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            self.cur_epoch_num = checkpoint['epoch_num']
            early_stop = checkpoint['early_stop']
            if early_stop.if_early_stop:
                print("Early stop!")
                self.optimizer_ = optimizer
                self.epoch_num = int(self.epoch_num) + int(self.cur_epoch_num)
                self.scheduler = scheduler
                self.early_stop = early_stop
                return self

        profile_iter = kwargs.get('profile_iter', None)
        profile_epoch = kwargs.get('profile_epoch', None)
        assert not (profile_iter and profile_epoch)

        if profile_epoch or profile_iter:  # Profile mode
            self.model.train()
            if profile_epoch:
                for epoch in range(int(profile_epoch)):
                    for i, data in enumerate(train_loader):
                        batch_x, batch_y = data[0], data[1]
                        masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
                        logits = self.model(batch_x.long().to(self.device), masks.to(self.device))
                        optimizer.zero_grad()
                        loss = loss_func(logits, batch_y.to(self.device))
                        loss.backward()
                        optimizer.step()
            else:
                num_iter = 0
                stop_flag = False
                for epoch in range(int(self.epoch_num)):
                    if stop_flag:
                        break
                    for i, data in enumerate(train_loader):
                        batch_x, batch_y = data[0], data[1]
                        masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
                        logits = self.model(batch_x.long().to(self.device), masks.to(self.device))
                        optimizer.zero_grad()
                        loss = loss_func(logits, batch_y.to(self.device))
                        loss.backward()
                        optimizer.step()
                        num_iter += 1
                        if num_iter > profile_iter:
                            stop_flag = True
                            break
            return self

        for epoch in range(int(self.cur_epoch_num), int(self.cur_epoch_num) + int(self.epoch_num)):
            self.model.train()
            # print('Current learning rate: %.5f' % optimizer.state_dict()['param_groups'][0]['lr'])
            epoch_avg_loss = 0
            epoch_avg_acc = 0
            val_avg_loss = 0
            val_avg_acc = 0
            num_train_samples = 0
            num_val_samples = 0
            for i, data in enumerate(train_loader):
                batch_x, batch_y = data[0], data[1]
                num_train_samples += len(batch_x)
                masks = torch.Tensor(np.array([[float(i != 0) for i in sample] for sample in batch_x]))
                logits = self.model(batch_x.long().to(self.device), masks.to(self.device))
                loss = loss_func(logits, batch_y.to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_avg_loss += loss.to('cpu').detach() * len(batch_x)
                prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                epoch_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

            epoch_avg_loss /= num_train_samples
            epoch_avg_acc /= num_train_samples
            # TODO: logger
            print('Epoch %d: Train loss %.4f, train acc %.4f' % (epoch, epoch_avg_loss, epoch_avg_acc))

            if val_loader is not None:
                self.model.eval()
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
                    if 'refit' not in mode:
                        early_stop.update(val_avg_loss)
                        if early_stop.if_early_stop:
                            self.early_stop_flag = True
                            print("Early stop!")
                            break

        scheduler.step()

        self.optimizer_ = optimizer
        self.epoch_num = int(self.epoch_num) + int(self.cur_epoch_num)
        self.scheduler = scheduler

        return self

    def predict_proba(self, dataset: Dataset, sampler=None, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS)
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
        assert sampler is None
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=None, num_workers=NUM_WORKERS)
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
            loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
        else:
            if not dataset.subset_sampler_used:
                loader = DataLoader(dataset=dataset.val_dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
            else:
                loader = DataLoader(dataset=dataset.train_for_val_dataset, batch_size=batch_size,
                                    sampler=dataset.val_sampler, num_workers=NUM_WORKERS)
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


class BaseODClassificationNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, optimizer, batch_size, epoch_num, lr_decay, weight_decay,
                 sgd_learning_rate=None, sgd_momentum=None, nesterov=None,
                 adam_learning_rate=None, beta1=None, random_state=None,
                 grayscale=False, device='cpu', **kwargs):
        super(BaseODClassificationNeuralNetwork, self).__init__()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epoch = epoch_num
        self.epoch_num = epoch_num
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_momentum = sgd_momentum
        self.nesterov = check_for_bool(nesterov)
        self.adam_learning_rate = adam_learning_rate
        self.beta1 = beta1
        self.random_state = random_state
        self.grayscale = grayscale
        self.model = None
        self.device = torch.device(device)
        self.time_limit = None
        self.load_path = None

        self.optimizer_ = None
        self.scheduler = None
        self.early_stop = None
        self.cur_epoch_num = 0

    def fit(self, dataset: DLDataset, mode='fit', **kwargs):
        assert self.model is not None

        if self.load_path:
            self.model.load_state_dict(torch.load(self.load_path))

        params = self.model.parameters()

        val_loader = None
        if 'refit' in mode:
            train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=NUM_WORKERS, collate_fn=dataset.train_dataset.collate_fn)
            if mode == 'refit_test':
                val_loader = DataLoader(dataset=dataset.test_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=NUM_WORKERS, collate_fn=dataset.test_dataset.collate_fn)
        else:
            train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=NUM_WORKERS, collate_fn=dataset.train_dataset.collate_fn)
            val_loader = DataLoader(dataset=dataset.val_dataset, batch_size=self.batch_size, shuffle=False,
                                    num_workers=NUM_WORKERS, collate_fn=dataset.val_dataset.collate_fn)
            # else:
            #     train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
            #                               sampler=dataset.train_sampler, num_workers=4,
            #                               collate_fn=dataset.train_dataset.collate_fn)
            #     val_loader = DataLoader(dataset=dataset.train_dataset, batch_size=self.batch_size,
            #                             sampler=dataset.val_sampler, num_workers=4,
            #                             collate_fn=dataset.train_dataset.collate_fn)

        if self.optimizer == 'SGD':
            optimizer = SGD(params=params, lr=self.sgd_learning_rate, momentum=self.sgd_momentum)
        elif self.optimizer == 'Adam':
            optimizer = Adam(params=params, lr=self.adam_learning_rate, betas=(self.beta1, 0.999))
        else:
            return ValueError("Optimizer %s not supported!" % self.optimizer)

        scheduler = MultiStepLR(optimizer, milestones=[int(self.max_epoch * 0.5), int(self.max_epoch * 0.75)],
                                gamma=self.lr_decay)
        early_stop = EarlyStop(patience=5, mode='min')

        if self.load_path:
            checkpoint = torch.load(self.load_path)
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            self.cur_epoch_num = checkpoint['epoch_num']
            early_stop = checkpoint['early_stop']
            if early_stop.if_early_stop:
                print("Early stop!")
                self.optimizer_ = optimizer
                self.epoch_num = int(self.epoch_num) + int(self.cur_epoch_num)
                self.scheduler = scheduler
                self.early_stop = early_stop
                return self

        profile_iter = kwargs.get('profile_iter', None)
        profile_epoch = kwargs.get('profile_epoch', None)
        assert not (profile_iter and profile_epoch)

        if profile_epoch or profile_iter:  # Profile mode
            self.model.train()
            if profile_epoch:
                for epoch in range(int(profile_epoch)):
                    for i, (_, batch_x, batch_y) in enumerate(train_loader):
                        loss, outputs = self.model(batch_x.float().to(self.device), batch_y.float().to(self.device))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            else:
                num_iter = 0
                stop_flag = False
                for epoch in range(int(self.epoch_num)):
                    if stop_flag:
                        break
                    for i, (_, batch_x, batch_y) in enumerate(train_loader):
                        loss, outputs = self.model(batch_x.float().to(self.device), batch_y.float().to(self.device))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        num_iter += 1
                        if num_iter > profile_iter:
                            stop_flag = True
                            break
            return self

        for epoch in range(int(self.cur_epoch_num), int(self.cur_epoch_num) + int(self.epoch_num)):
            self.model.train()
            # print('Current learning rate: %.5f' % optimizer.state_dict()['param_groups'][0]['lr'])
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
                self.model.eval()
                with torch.no_grad():
                    for i, (_, batch_x, batch_y) in enumerate(val_loader):
                        loss, outputs = self.model(batch_x.float().to(self.device), batch_y.float().to(self.device))
                        val_avg_loss += loss.to('cpu').detach() * len(batch_x)
                        num_val_samples += len(batch_x)

                    val_avg_loss /= num_val_samples
                    print('Epoch %d: Val loss %.4f' % (epoch, val_avg_loss))

                    # Early stop
                    if 'refit' not in mode:
                        early_stop.update(val_avg_loss)
                        if early_stop.if_early_stop:
                            self.early_stop_flag = True
                            print("Early stop!")
                            break

        self.optimizer_ = optimizer
        self.epoch_num = int(self.epoch_num) + int(self.cur_epoch_num)
        self.scheduler = scheduler

        return self

    def predict(self, dataset: Dataset, sampler=None, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=NUM_WORKERS, collate_fn=dataset.collate_fn)
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
