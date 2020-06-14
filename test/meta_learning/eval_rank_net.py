import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from solnml.components.meta_learning.ranknet import LambdaRankNN

# generate query data
X = np.array([[0.2, 0.3, 0.4],
              [0.1, 0.7, 0.4],
              [0.3, 0.4, 0.1],
              [0.8, 0.4, 0.3],
              [0.9, 0.35, 0.25]])
y = np.array([0, 1, 0, 0, 2])
qid = np.array([1, 1, 1, 2, 2])

# train model
ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(X, y, qid, epochs=100)
y_pred = ranker.predict(X)
print(y_pred)
ranker.evaluate(X, y, qid, eval_at=2)
