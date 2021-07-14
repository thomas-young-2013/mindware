import os
import sys
import numpy as np
import pickle as pk
sys.path.append(os.getcwd())
from mindware.datasets.utils import calculate_metafeatures
from mindware.components.utils.constants import MULTICLASS_CLS

datasets = ['balloon', 'kc1', 'quake', 'segment', 'madelon', 'space_ga',
            'kr-vs-kp', 'cpu_small', 'sick', 'hypothyroid(1)', 'hypothyroid(2)',
            'pollen', 'analcatdata_supreme', 'splice', 'abalone', 'spambase',
            'winequality_white', 'waveform-5000(1)', 'waveform-5000(2)',
            'page-blocks(1)', 'page-blocks(2)', 'cpu_act', 'optdigits',
            'satimage', 'wind', 'musk', 'delta_ailerons', 'bank32nh',
            'mushroom', 'puma8NH']

task_ids = list()
X = list()
result = dict()
sorted_keys = None

for _dataset in datasets:
    print('Creating embedding for dataset - %s.' % _dataset)
    # Calculate metafeature for datasets.
    try:
        feature_dict = calculate_metafeatures(_dataset, task_type=MULTICLASS_CLS)
    except Exception as e:
        print(e)
        continue
    sorted_keys = sorted(feature_dict.keys())
    meta_instance = [feature_dict[key] for key in sorted_keys]

    X.append(meta_instance)
    task_ids.append(_dataset)

result['task_ids'] = task_ids
result['dataset_embedding'] = np.array(X)
result['meta_feature_names'] = sorted_keys

with open('dataset_metafeatures.pkl', 'wb') as f:
    pk.dump(result, f)
