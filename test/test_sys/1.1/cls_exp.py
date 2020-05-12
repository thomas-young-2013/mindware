import argparse
import os
import sys
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())
from solnml.datasets.utils import load_train_test_data
from solnml.estimators import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pc4')
parser.add_argument('--time_limit', type=int, default=1200)
parser.add_argument('--eval_type', type=str, default='holdout', choices=['holdout', 'cv', 'partial'])
parser.add_argument('--ens_method', default='ensemble_selection',
                    choices=['none', 'bagging', 'blending', 'stacking', 'ensemble_selection'])
parser.add_argument('--n_jobs', type=int, default=1)

args = parser.parse_args()

dataset = args.dataset
time_limit = args.time_limit
eval_type = args.eval_type
n_jobs = args.n_jobs
ensemble_method = args.ens_method
if ensemble_method == 'none':
    ensemble_method = None

save_dir = './data/eval_exps/soln-ml'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print('==> Start to evaluate with Budget %d' % time_limit)

train_data, test_data = load_train_test_data(dataset)

clf = Classifier(time_limit=time_limit,
                 output_dir=save_dir,
                 ensemble_method=ensemble_method,
                 evaluation=eval_type,
                 metric='acc',
                 n_jobs=n_jobs)
clf.fit(train_data)
pred = clf.predict(test_data)
print(accuracy_score(test_data.data[1], pred))
