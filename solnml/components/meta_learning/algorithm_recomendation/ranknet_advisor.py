import os
import numpy as np
import pickle as pk
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Dense, Input, Subtract
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from solnml.utils.logging_utils import get_logger
from solnml.components.meta_learning.algorithm_recomendation.base_advisor import BaseAdvisor


class RankNetAdvisor(BaseAdvisor):
    def __init__(self, n_algorithm=3,
                 task_type=None,
                 metric='acc',
                 rep=3,
                 total_resource=20,
                 exclude_datasets=None,
                 meta_dir=None):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        super().__init__(n_algorithm, task_type, metric, rep, total_resource,
                         'ranknet', exclude_datasets, meta_dir)
        self.model = None

    @staticmethod
    def create_pairwise_data(X, y):
        X1, X2, labels = list(), list(), list()
        for _X, _y in zip(X, y):
            n_sample = len(_X)
            for i in range(n_sample):
                for j in range(i+1, n_sample):
                    if np.isnan(_X[i]).any() or np.isnan(_X[j]).any():
                        continue
                    X1.append(_X[i])
                    X1.append(_X[j])
                    X2.append(_X[j])
                    X2.append(_X[i])
                    _label = 1 if _y[i] > _y[j] else 0
                    labels.append(_label)
                    labels.append(1 - _label)
        X1, X2, labels = np.asarray(X1), np.asarray(X2), np.asarray(labels)
        # perm = np.random.permutation(X1.shape[0])
        # return X1[perm], X2[perm], labels[perm]
        return X1, X2, labels

    @staticmethod
    def create_model(input_shape, hidden_layer_sizes, activation, solver):
        """
        Build Keras Ranker NN model (Ranknet / LambdaRank NN).
        """
        # Neural network structure
        hidden_layers = list()
        hidden_layers.append(BatchNormalization())

        for i in range(len(hidden_layer_sizes)):
            hidden_layers.append(
                Dense(hidden_layer_sizes[i], activation=activation[i], name=str(activation[i]) + '_layer' + str(i)))
        h0 = Dense(1, activation='linear', name='Identity_layer')
        input1 = Input(shape=(input_shape,), name='Input_layer1')
        input2 = Input(shape=(input_shape,), name='Input_layer2')
        x1 = input1
        x2 = input2
        for i in range(len(hidden_layer_sizes)):
            x1 = hidden_layers[i](x1)
            x2 = hidden_layers[i](x2)
        x1 = h0(x1)
        x2 = h0(x2)
        # Subtract layer
        subtracted = Subtract(name='Subtract_layer')([x1, x2])
        # sigmoid
        out = Activation('sigmoid', name='Activation_layer')(subtracted)
        # build model
        model = Model(inputs=[input1, input2], outputs=out)

        # categorical_hinge, binary_crossentropy
        # sgd = SGD(lr=0.3, momentum=0.9, decay=0.001, nesterov=False)
        model.compile(optimizer=solver, loss="categorical_hinge", metrics=['accuracy'])
        return model

    def fit(self, **kwargs):
        _X, _y = self.load_train_data()
        X1, X2, y = self.create_pairwise_data(_X, _y)

        l1_size = kwargs.get('layer1_size', 64)
        l2_size = kwargs.get('layer2_size', 32)
        act_func = kwargs.get('activation', 'relu')
        batch_size = kwargs.get('batch_size', 32)

        self.model = self.create_model(X1.shape[1], hidden_layer_sizes=(l1_size, l2_size,),
                                       activation=(act_func, act_func,),
                                       solver='adam')

        self.model.fit([X1, X2], y, epochs=200, batch_size=batch_size)

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

        X = self.load_test_data(dataset_meta_feat)
        ranker_output = K.function([self.model.layers[0].input], [self.model.layers[-3].get_output_at(0)])
        return ranker_output([X])[0].ravel()
