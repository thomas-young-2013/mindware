import argparse
import os
import sys
import time

from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from solnml.utils.data_manager import DataManager
from solnml.estimators import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=1200)
parser.add_argument('--eval_type', type=str, default='holdout', choices=['holdout', 'cv', 'partial'])
parser.add_argument('--ens_method', default='ensemble_selection',
                    choices=['none', 'bagging', 'blending', 'stacking', 'ensemble_selection'])
parser.add_argument('--n_jobs', type=int, default=1)

args = parser.parse_args()

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

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
dm = DataManager(X_train, y_train)
train_data = dm.get_data_node(X_train, y_train)
test_data = dm.get_data_node(X_test, y_test)

clf = Classifier(time_limit=time_limit,
                 output_dir=save_dir,
                 ensemble_method=ensemble_method,
                 evaluation=eval_type,
                 metric='acc',
                 n_jobs=n_jobs)
clf.fit(train_data)
pred = clf.predict(test_data)
print(balanced_accuracy_score(test_data.data[1], pred))
