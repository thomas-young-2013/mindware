import os
import argparse

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mindware.utils.data_manager import DataManager
from mindware.estimators import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=150)
parser.add_argument('--eval_type', type=str, default='holdout', choices=['holdout', 'cv', 'partial'])
parser.add_argument('--ens_method', default='ensemble_selection',
                    choices=['none', 'bagging', 'blending', 'stacking', 'ensemble_selection'])

args = parser.parse_args()
time_limit = args.time_limit
eval_type = args.eval_type
ensemble_method = args.ens_method

save_dir = './logs/tmps/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def evaluate():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    try:
        dm = DataManager(X_train, y_train)
        train_data = dm.get_data_node(X_train, y_train)
        test_data = dm.get_data_node(X_test, y_test)

        clf = Classifier(dataset_name='iris',
                         time_limit=150,
                         output_dir=save_dir,
                         ensemble_method=ensemble_method,
                         evaluation=eval_type,
                         metric='acc')
        clf.fit(train_data)
        clf.refit()
        pred = clf.predict(test_data)
        print('final score', clf.score(test_data))
    except Exception as e:
        return False
    return True
