import os
import numpy as np
import pickle as pk

from torch import nn, optim, from_numpy
import torch
from torch.utils.data import Dataset, DataLoader

from solnml.utils.logging_utils import get_logger
from solnml.components.meta_learning.algorithm_recomendation.base_advisor import BaseAdvisor


class CategoricalHingeLoss(nn.Module):
    def forward(self, input, target):
        pos = (1. - target) * (1. - input) + target * input
        neg = target * (1. - input) + (1. - target) * input
        return torch.sum(torch.max(torch.zeros_like(neg - pos + 1.), neg - pos + 1.)) / len(input)


class PairwiseDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1_array, self.X2_array, self.y_array = X1, X2, y.reshape(y.shape[0], 1)

    def __getitem__(self, index):
        data1 = from_numpy(self.X1_array[index]).float()
        data2 = from_numpy(self.X2_array[index]).float()
        y_true = from_numpy(self.y_array[index]).float()
        return data1, data2, y_true

    def __len__(self):
        return self.X1_array.shape[0]


class RankNet(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, activation):
        super(RankNet, self).__init__()
        self.model = nn.Sequential()
        self.input_shape = input_shape
        self.output_sigmoid = nn.Sigmoid()
        self.model.add_module('BatchNorm', nn.BatchNorm1d(input_shape))
        self.model.add_module('linear_' + str(hidden_layer_sizes[0]), nn.Linear(input_shape, hidden_layer_sizes[0]))
        self.model.add_module('act_func_' + str(0), nn.ReLU(inplace=True))
        for i in range(1, len(hidden_layer_sizes)):
            self.model.add_module('linear_' + str(hidden_layer_sizes[i]),
                                  nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
            self.model.add_module('act_func_' + str(i),
                                  nn.ReLU(inplace=True))  # TODO: change ReLU to customized act_func
        self.model.add_module('output', nn.Linear(hidden_layer_sizes[-1], 1))

    def forward(self, input1, input2):
        s1 = self.model(input1)
        s2 = self.model(input2)
        return self.output_sigmoid(s1 - s2)

    def predict(self, input):
        return self.model(input).detach()


class RankNetAdvisor(BaseAdvisor):
    def __init__(self,
                 rep=3,
                 metric='acc',
                 n_algorithm=3,
                 task_type=None,
                 total_resource=1200,
                 exclude_datasets=None,
                 meta_dir=None):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        super().__init__(n_algorithm, task_type, metric, rep, total_resource,
                         'ranknet', exclude_datasets, meta_dir)
        self.model = None

    @staticmethod
    def create_pairwise_data(X, y):
        X1, X2, labels = list(), list(), list()
        n_algo = y.shape[1]

        for _X, _y in zip(X, y):
            if np.isnan(_X).any():
                continue
            meta_vec = _X
            for i in range(n_algo):
                for j in range(i + 1, n_algo):
                    if (_y[i] == -1) or (_y[j] == -1):
                        continue

                    vector_i, vector_j = np.zeros(n_algo), np.zeros(n_algo)
                    vector_i[i] = 1
                    vector_j[j] = 1

                    meta_x1 = list(meta_vec.copy())
                    meta_x1.extend(vector_i.copy())

                    meta_x2 = list(meta_vec.copy())
                    meta_x2.extend(vector_j.copy())

                    X1.append(meta_x1)
                    X1.append(meta_x2)
                    X2.append(meta_x2)
                    X2.append(meta_x1)
                    _label = 1 if _y[i] > _y[j] else 0
                    labels.append(_label)
                    labels.append(1 - _label)
        return np.asarray(X1), np.asarray(X2), np.asarray(labels)

    @staticmethod
    def create_model(input_shape, hidden_layer_sizes, activation):
        return RankNet(input_shape, hidden_layer_sizes, activation)

    def fit(self, **kwargs):
        l1_size = kwargs.get('layer1_size', 256)
        l2_size = kwargs.get('layer2_size', 64)
        act_func = kwargs.get('activation', 'tanh')
        batch_size = kwargs.get('batch_size', 256)
        epochs = 200

        _X, _y, _ = self.metadata_manager.load_meta_data()
        X1, X2, y = self.create_pairwise_data(_X, _y)

        train_data = PairwiseDataset(X1, X2, y)
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        self.input_shape = X1.shape[1]


        self.model = RankNet(X1.shape[1], (l1_size, l2_size,), (act_func, act_func,))
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        loss_fun = CategoricalHingeLoss()
        self.model.train()

        for epoch in range(epochs):
            train_loss = 0
            train_samples = 0
            train_acc = 0
            for i, (data1, data2, y_true) in enumerate(train_loader):
                optimizer.zero_grad()
                y_pred = self.model(data1, data2)
                loss = loss_fun(y_pred, y_true)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(data1)
                train_samples += len(data1)

                train_acc += sum(y_pred.detach().numpy().round() == y_true.detach().numpy())

            print('Epoch{}, loss : {}'.format(epoch, train_loss / len(train_data)))
            print('Epoch{}, acc : {}'.format(epoch, train_acc / len(train_data)))

    def predict(self, dataset_meta_feat):
        meta_learner_filename = self.meta_dir + 'ranknet_model_%s_%s_%s.pkl' % (
            self.meta_algo, self.metric, self.hash_id)

        if self.model is None:
            if os.path.exists(meta_learner_filename):
                print('Load model from file: %s.' % meta_learner_filename)
                with open(meta_learner_filename, 'rb') as f:
                    self.model = pk.load(f)
            else:
                self.fit()
                with open(meta_learner_filename, 'wb') as f:
                    pk.dump(self.model, f)
                print('Dump model to file: %s.' % meta_learner_filename)

        n_algo = self.n_algo_candidates
        _X = list()
        for i in range(n_algo):
            vector_i = np.zeros(n_algo)
            vector_i[i] = 1
            _X.append(list(dataset_meta_feat.copy()) + list(vector_i))
        X = np.asarray(_X)

        X = from_numpy(X).float()
        self.model.eval()
        pred = self.model.predict(X).numpy()
        return pred.ravel()
