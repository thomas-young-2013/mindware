import os
import numpy as np
import pickle as pk
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Activation, Dense, Input, Subtract
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from solnml.utils.logging_utils import get_logger
from solnml.components.meta_learning.algorithm_recomendation.base_advisor import BaseAdvisor


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
        _X, _y, _ = self.metadata_manager.load_meta_data()
        X1, X2, y = self.create_pairwise_data(_X, _y)

        l1_size = kwargs.get('layer1_size', 256)
        l2_size = kwargs.get('layer2_size', 128)
        act_func = kwargs.get('activation', 'tanh')
        batch_size = kwargs.get('batch_size', 128)

        meta_learner_filename = os.path.join(self.meta_dir, "meta_learner", 'ranknet_model_%s_%s_%s.h5' % (
            self.meta_algo, self.metric, self.hash_id))
        if os.path.exists(meta_learner_filename):
            # print("load model...")
            self.model = load_model(meta_learner_filename)
        else:
            # print("fit model..")
            self.model = self.create_model(X1.shape[1], hidden_layer_sizes=(l1_size, l2_size,),
                                           activation=(act_func, act_func,),
                                           solver='adam')

            self.model.fit([X1, X2], y, epochs=200, batch_size=batch_size)
            # print("save model...")
            self.model.save(meta_learner_filename)

    def predict(self, dataset_meta_feat):
        n_algo = self.n_algo_candidates
        _X = list()
        for i in range(n_algo):
            vector_i = np.zeros(n_algo)
            vector_i[i] = 1
            item = list(dataset_meta_feat.copy()) + list(vector_i)
            _X.append(item)
        X = np.asarray(_X)
        ranker_output = K.function([self.model.layers[0].input], [self.model.layers[-3].get_output_at(0)])
        return ranker_output([X])[0].ravel()
