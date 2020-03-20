import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from automlToolkit.components.feature_engineering.transformations.preprocessor.image2vector import \
    Image2VectorTransformation
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.utils.constants import *
from automlToolkit.components.utils.image_util import *

x = []
for i in range(300):
    x.append(np.array([i, np.random.randint(0, 256, (256, 256, 3)), 2]))
y = [0, 1]
x = np.array(x)
i2v = Image2VectorTransformation()
data = (x, y)
feature_type = [NUMERICAL, IMAGE, DISCRETE]
datanode = DataNode(data, feature_type)
print(i2v.operate(datanode))
