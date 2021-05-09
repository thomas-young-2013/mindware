import os
import shutil
from sklearn.datasets import load_iris, load_boston
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from solnml.utils.data_manager import DataManager
from solnml.estimators import Classifier, Regressor


def test_cls():
    save_dir = './data/eval_exps/soln-ml'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_limit = 60
    print('==> Start to evaluate with Budget %d' % time_limit)
    ensemble_method = 'blending'
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
    clf.fit(train_data)
    print(clf.summary())

    pred = clf.predict(test_data)
    print(accuracy_score(test_data.data[1], pred))

    shutil.rmtree(save_dir)


def test_rgs():
    time_limit = 60
    print('==> Start to evaluate with Budget %d' % time_limit)
    ensemble_method = 'blending'
    eval_type = 'holdout'

    boston = load_boston()
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    dm = DataManager(X_train, y_train)
    train_data = dm.get_data_node(X_train, y_train)
    test_data = dm.get_data_node(X_test, y_test)

    save_dir = './data/eval_exps/soln-ml'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rgs = Regressor(metric='mse',
                    ensemble_method=ensemble_method,
                    enable_meta_algorithm_selection=False,
                    evaluation=eval_type,
                    time_limit=time_limit,
                    output_dir=save_dir)

    rgs.fit(train_data)
    print(rgs.summary())

    pred = rgs.predict(test_data)
    print(mean_squared_error(test_data.data[1], pred))

    shutil.rmtree(save_dir)


if __name__ == '__main__':
    test_cls()
    test_rgs()
