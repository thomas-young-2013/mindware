import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Dense, Input, Subtract
from keras.layers import InputLayer, Dropout, BatchNormalization
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
        return np.asarray(X1), np.asarray(X2), np.asarray(labels)

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
        model.compile(optimizer=solver, loss="categorical_hinge", metrics=['acc'])
        return model

    def fit(self):
        _X, _y = self.load_train_data()
        X1, X2, y = self.create_pairwise_data(_X, _y)

        self.model = self.create_model(X1.shape[1], hidden_layer_sizes=(64, 32,),
                                       activation=('relu', 'relu',),
                                       solver='adam')

        self.model.fit([X1, X2], y, epochs=200, batch_size=64)

    def predict(self, dataset_meta_feat):
        X = self.load_test_data(dataset_meta_feat)
        ranker_output = K.function([self.model.layers[0].input], [self.model.layers[-3].get_output_at(0)])
        return ranker_output([X])[0].ravel()
