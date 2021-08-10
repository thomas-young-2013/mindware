import os
import sys
import argparse

from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from mindware.utils.data_manager import DataManager
from mindware.estimators import Classifier
from mindware.distrib.worker import EvaluationWorker
from mindware.distrib.master import Master

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=300)
parser.add_argument('--eval_type', type=str, default='holdout',
                    choices=['holdout', 'cv', 'partial'])
parser.add_argument('--ens_method', default='ensemble_selection',
                    choices=['none', 'bagging', 'blending', 'stacking', 'ensemble_selection'])
parser.add_argument('--n_jobs', type=int, default=1)
parser.add_argument('--role', type=str, default='master',
                    choices=['master', 'worker'])
parser.add_argument('--n_workers', type=int)
parser.add_argument('--parallel_strategy', type=str, default='async',
                    choices=['sync', 'async'])
parser.add_argument('--master_ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=13579)

args = parser.parse_args()
time_limit = args.time_limit
eval_type = args.eval_type
n_jobs = args.n_jobs
ensemble_method = args.ens_method
role = args.role

if ensemble_method == 'none':
    ensemble_method = None

save_dir = './data/mindware'
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
                 enable_meta_algorithm_selection=False,
                 n_jobs=n_jobs)

clf.initialize(train_data, tree_id=0)

if role == 'master':
    # bind the IP, port, etc.
    master = Master(clf, ip=args.master_ip, port=args.port)
    master.run()
else:
    # Set up evaluation workers.
    evaluator = clf.get_evaluator()
    worker = EvaluationWorker(evaluator, args.master_ip, args.port)
    worker.run()
