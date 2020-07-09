import os
import sys
import argparse

sys.path.append(os.getcwd())

from solnml.datasets.image_dataset import ImageDataset
from solnml.estimators import ImageClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=1)
args = parser.parse_args()
n_jobs = args['n_jobs']
print('n_jobs is set to %d.' % n_jobs)

data_dir = 'data/img_datasets/cifar10/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True)
clf = ImageClassifier(time_limit=1800,
                      include_algorithms=['mobilenet'],
                      evaluation='partial',
                      image_size=32,
                      ensemble_method='ensemble_selection',
                      n_jobs=n_jobs)
clf.fit(image_data)
image_data.set_test_path(data_dir)
print(clf.predict_proba(image_data))
pred = clf.predict(image_data)
