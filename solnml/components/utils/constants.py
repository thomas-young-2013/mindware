"""
Constants used in feature types
"""
feature_types = ['discrete', 'numerical', 'ordinal', 'categorical', 'text']
DISCRETE = 'discrete'
NUMERICAL = 'numerical'
TEXT = 'text'
CATEGORICAL = 'categorical'
ORDINAL = 'ordinal'
TEXT_EMBEDDING = 'text_output'
IMAGE = 'image'
IMAGE_EMBEDDING = 'image_output'

FEATURE_TYPES = [DISCRETE, NUMERICAL, TEXT, CATEGORICAL, ORDINAL, TEXT_EMBEDDING, IMAGE, IMAGE_EMBEDDING]

"""
Constants used in return value
"""

SUCCESS = 0
TIMEOUT = 1
ERROR = 2
MEMORYOUT = 3
CRASHED = 4

"""
Constant used in task type
"""

CLASSIFICATION = 0
BINARY_CLS = 1
MULTICLASS_CLS = 2
MULTILABEL_CLS = 3
REGRESSION = 4
IMG_CLS = 5
TEXT_CLS = 6

type_dict = {'binary': BINARY_CLS,
             'multiclass': MULTICLASS_CLS,
             'multilabel-indicator': MULTILABEL_CLS,
             'continuous': REGRESSION
             }

REG_TASKS = [REGRESSION]
CLS_TASKS = [CLASSIFICATION, BINARY_CLS, MULTICLASS_CLS, MULTILABEL_CLS, IMG_CLS, TEXT_CLS]

TASK_TYPES = REG_TASKS + CLS_TASKS

TASK_TYPE2STR = {BINARY_CLS: "binary",
                 MULTICLASS_CLS: "multiclass",
                 REGRESSION: "regression"}

TASK_STR2TYPE = {"binary": BINARY_CLS,
                 "multiclass": MULTICLASS_CLS,
                 "regression": REGRESSION}

DENSE = 5
SPARSE = 6
PREDICTIONS = 7
INPUT = 8

SIGNED_DATA = 9
UNSIGNED_DATA = 10
