import argparse
import os
import sys

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from automlToolkit.utils.data_manager import DataManager
from automlToolkit.estimators import Regressor

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=1200)
args = parser.parse_args()

time_limit = args.time_limit

print('==> Start to evaluate with Budget %d' % time_limit)

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
dm = DataManager(X_train, y_train)
train_data = dm.get_data_node(X_train, y_train)
test_data = dm.get_data_node(X_test, y_test)

save_dir = './data/eval_exps/automl-toolkit'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

rgs = Regressor(
    metric='mse',
    time_limit=time_limit,
    output_dir=save_dir,
    random_state=1)

rgs.fit(train_data)
pred = rgs.predict(test_data)

print(mean_squared_error(test_data.data[1], pred))
