import os
import shutil
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from solnml.utils.data_manager import DataManager
from solnml.estimators import Classifier


def test_cls():
    save_dir = './data/eval_exps/soln-ml'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_limit = 60
    print('==> Start to evaluate with Budget %d' % time_limit)
    ensemble_method = 'ensemble_selection'
    eval_type = 'holdout'

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
    dm = DataManager(X_train, y_train)
    train_data = dm.get_data_node(X_train, y_train)
    test_data = dm.get_data_node(X_test, y_test)

    clf = Classifier(time_limit=time_limit,
                     output_dir=save_dir,
                     ensemble_method=ensemble_method,
                     enable_meta_algorithm_selection=False,
                     ensemble_size=10,
                     evaluation=eval_type,
                     metric='acc')
    clf.fit(train_data, tree_id=2)
    print(clf.summary())

    pred = clf.predict(test_data)
    print(accuracy_score(test_data.data[1], pred))

    shutil.rmtree(save_dir)


if __name__ == '__main__':
    test_cls()
