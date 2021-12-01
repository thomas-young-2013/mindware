from mindware.datasets.base_dl_dataset import DLDataset
from mindware.components.models.base_nn import BaseImgClassificationNeuralNetwork
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from mindware.components.models.search.nas_utils.config2net import get_net_from_config
from mindware.components.models.search.nas_utils.nb1_utils import bench101_opt_choices

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import RMSprop
from torch.optim.lr_scheduler import MultiStepLR
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

NUM_WORKERS = 10


class NASBench101Searcher(BaseImgClassificationNeuralNetwork):
    def __init__(self, arch_config, batch_size=128,
                 epoch_num=108, learning_rate=0.2,
                 momentum=0.9, weight_decay=1e-4,
                 lr_decay=1e-1, random_state=None,
                 grayscale=False, device='cpu', **kwargs):
        self.arch_config = arch_config
        self.batch_size = batch_size
        self.max_epoch = epoch_num
        self.epoch_num = epoch_num
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.random_state = random_state
        self.grayscale = grayscale
        self.model = None
        self.device = torch.device(device)
        self.time_limit = None
        self.load_path = None

        self.optimizer_ = None
        self.scheduler = None
        self.cur_epoch_num = 0

    def set_empty_model(self, config, dataset):
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        self.model = get_net_from_config(space='nasbench101', config=config, n_classes=len(dataset.classes))
        self.model.to(self.device)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'NB101Searcher',
                'name': 'NASBench101Searcher',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    def fit(self, dataset: DLDataset, mode='fit', **kwargs):
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        self.model = get_net_from_config(space='nasbench101', config=self.arch_config, n_classes=len(dataset.classes))
        self.model.to(self.device)

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

        optimizer = RMSprop(params=params, lr=self.learning_rate, momentum=self.momentum,
                            weight_decay=self.weight_decay)

        scheduler = MultiStepLR(optimizer, milestones=[int(self.max_epoch * 0.5), int(self.max_epoch * 0.75)],
                                gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()

        if self.load_path:
            checkpoint = torch.load(self.load_path)
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            self.cur_epoch_num = checkpoint['epoch_num']

        for epoch in range(int(self.cur_epoch_num), int(self.cur_epoch_num) + int(self.epoch_num)):
            self.model.train()
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

            scheduler.step()

        self.optimizer_ = optimizer
        self.epoch_num = int(self.epoch_num) + int(self.cur_epoch_num)
        self.scheduler = scheduler

        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        vertex = 7
        cs = ConfigurationSpace()
        total_edges = sum(list(range(vertex)))
        for e in range(0, total_edges):
            cs.add_hyperparameter(CategoricalHyperparameter('edge_%d' % e, choices=[0, 1]))

        for v in range(1, vertex - 1):
            cs.add_hyperparameter(CategoricalHyperparameter('vertex_%d' % v, choices=bench101_opt_choices))

        return cs
