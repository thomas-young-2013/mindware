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

SUCCESS = 0
TIMEOUT = 1
ERROR = 2
MEMORYOUT = 3
CRASHED = 4

CLASSIFICATION = 'classification'
REGRESSION = 'regression'