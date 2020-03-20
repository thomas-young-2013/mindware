import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from automlToolkit.components.feature_engineering.transformations.preprocessor.text2vector import \
    Text2VectorTransformation
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.utils.constants import *

x = np.array([[1, 'I am good', 'I am right', 3], [2, 'He is silly', 'He is stupid', 4]])
y = np.array([0, 1])

t2v = Text2VectorTransformation()
data = (x, y)
feature_type = [NUMERICAL, TEXT, TEXT, DISCRETE]
datanode = DataNode(data, feature_type)
print(t2v.operate(datanode))