import os
import sys
import argparse

sys.path.append(os.getcwd())

from solnml.datasets.image_dataset import ImageDataset
from solnml.estimators import ImageClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=3)
parser.add_argument('--dataset', type=str, default='extremely_small')
args = parser.parse_args()
n_jobs = args.n_jobs
dataset = args.dataset
print('n_jobs is set to %d.' % n_jobs)

data_dir = 'data/img_datasets/%s/' % dataset
image_data = ImageDataset(data_path=data_dir, train_val_split=True)
clf = ImageClassifier(time_limit=10000000,
                      include_algorithms=['mobilenet', 'resnet34', 'efficientnet'],
                      evaluation='partial',
                      image_size=32,
                      ensemble_method='ensemble_selection',
                      skip_profile=True,
                      n_jobs=n_jobs)
clf.fit(image_data)
image_data.set_test_path(data_dir)
print(clf.predict_proba(image_data))
pred = clf.predict(image_data)
