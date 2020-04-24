import argparse
import os
import sys

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from automlToolkit.utils.data_manager import DataManager
from automlToolkit.components.feature_engineering.fe_pipeline import FEPipeline
from automlToolkit.estimators import Regressor

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

print('==> Start to evaluate with Budget %d' % time_limit)

dm = DataManager()
train_node = dm.load_train_csv("train_dataset.csv", label_col=-1, header='infer', na_values=['nan', '?'])
test_node = dm.load_test_csv("test_dataset.csv", header='infer', has_label=True)
from automlToolkit.components.utils.constants import REGRESSION

pipeline = FEPipeline(fe_enabled=False, task_type=REGRESSION)
train_data = pipeline.fit_transform(train_node)
test_data = pipeline.transform(test_node)

save_dir = './data/eval_exps/automl-toolkit'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

rgs = Regressor(metric='mse',
                ensemble_method=ensemble_method,
                evaluation=eval_type,
                time_limit=time_limit,
                output_dir=save_dir,
                random_state=1,
                n_jobs=n_jobs)

rgs.fit(train_data)
pred = rgs.predict(test_data)

print(mean_squared_error(test_data.data[1], pred))
