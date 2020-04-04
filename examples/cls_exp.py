import argparse
import os
import sys
import time

from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from automlToolkit.utils.data_manager import DataManager
from automlToolkit.estimators import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=1200)
args = parser.parse_args()

time_limit = args.time_limit

print('==> Start to evaluate with Budget %d' % time_limit)

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
dm = DataManager(X_train, y_train)
train_data = dm.get_data_node(X_train, y_train)
test_data = dm.get_data_node(X_test, y_test)

save_dir = './data/eval_exps/automl-toolkit'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
clf = Classifier(time_limit=time_limit, output_dir=save_dir, random_state=1, metric='acc', n_jobs=1)
_start_time = time.time()
_iter_id = 0

clf.fit(train_data)
pred = clf.predict(test_data)

print(balanced_accuracy_score(test_data.data[1], pred))
