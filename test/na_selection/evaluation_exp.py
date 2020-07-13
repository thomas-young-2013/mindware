import os
import sys
import time
import argparse
import pickle as pk
from sklearn.metrics import accuracy_score
sys.path.append(os.getcwd())

from solnml.datasets.image_dataset import ImageDataset
from solnml.estimators import ImageClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=3)
parser.add_argument('--time_limit', type=int, default=36000)
parser.add_argument('--max_epoch', type=int, default=81)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--opt_method', type=str, default='ours')
parser.add_argument('--eval', type=str, default='partial')
networks_template = ['mobilenet', 'resnet34', 'efficientnet',
                     'resnet50', 'resnet152', 'resnet101',
                     'densenet121']
parser.add_argument('--networks', type=str, default=','.join(networks_template))
args = parser.parse_args()
n_jobs = args.n_jobs
dataset = args.dataset
networks = args.networks.split(',')
time_limit = args.time_limit
opt_method = args.opt_method
evaluation = args.eval
max_epoch = args.max_epoch
print('n_jobs is set to %d.' % n_jobs)
print('networks included', networks)

data_dir = 'data/img_datasets/%s/' % dataset
image_data = ImageDataset(data_path=data_dir, train_val_split=True)
clf = ImageClassifier(time_limit=time_limit,
                      include_algorithms=networks,
                      evaluation=evaluation,
                      image_size=32,
                      ensemble_method=None,
                      skip_profile=True,
                      max_epoch=max_epoch,
                      n_jobs=n_jobs)
image_data.set_test_path(data_dir)

if opt_method == 'ours':
    clf.fit(image_data)
else:
    clf.fit(image_data, opt_method=opt_method)
res = clf.get_runtime_history()

preds = clf.predict(image_data, mode='val')
val_labels = image_data.get_labels(mode='val')
metric = accuracy_score
val_acc = metric(val_labels, preds)
print('validation acc', val_acc)

preds = clf.predict(image_data, mode='test')
test_labels = image_data.get_labels(mode='test')
test_acc = metric(test_labels, preds)
print('test acc', test_acc)

file_path = 'autodl_%s_%s_%d_%d.pkl' % (dataset, opt_method, time_limit, time.time())
result = [res, val_acc, test_acc]
print(res)
with open(file_path, 'wb') as f:
    pk.dump(result, f)
