import os


from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from solnml.utils.data_manager import DataManager
from solnml.estimators import Regressor


def main():
    ensemble_method = None
    time_limit = 300
    print('==> Start to evaluate with Budget %d' % time_limit)
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
                    evaluation=eval_type,
                    time_limit=time_limit,
                    output_dir=save_dir)

    rgs.fit(train_data)
    pred = rgs.predict(test_data)

    print(mean_squared_error(test_data.data[1], pred))


if __name__ == '__main__':
    main()
