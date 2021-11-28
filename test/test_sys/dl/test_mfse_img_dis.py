import os
import sys
import time
import shutil
from torchvision import transforms
from sklearn.metrics import accuracy_score
import argparse

sys.path.append(os.getcwd())

from mindware.datasets.image_dataset import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--role', type=str, choices=['master', 'worker'])
parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=int, default=13579)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_workers', type=int, default=2)

args = parser.parse_args()
role = args.role
ip = args.ip
port = args.port
R = args.R
eta = args.eta
n_workers = args.n_workers

# data_dir = 'data/img_datasets/extremely_small/'
data_dir = 'data/img_datasets/cifar10/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True)
save_dir = './data/eval_exps/mindware'

from mindware.autodl import AutoDL

max_epoch = 81
clf = AutoDL(time_limit=18000,
             max_epoch=max_epoch,
             include_algorithms=['resnet32_32'],
             ensemble_method='ensemble_selection',
             evaluation='partial',
             skip_profile=True,
             output_dir=save_dir,
             n_jobs=2)

# Space
config_space = clf.get_pipeline_config_space(clf.include_algorithms)

# Setting
timestamp = 5
save_dir = os.path.join(save_dir, 'mindware_%s' % timestamp)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)

# Evaluator
from mindware.components.utils.worker import async_mqmfWorker as Worker
from mindware.components.evaluators.dl_evaluator import DLEvaluator
from sklearn.metrics._scorer import accuracy_scorer

evaluator = DLEvaluator(task_type=5,
                        max_epoch=max_epoch,
                        scorer=accuracy_scorer,
                        dataset=image_data,
                        device='cuda:0',
                        continue_training=False,
                        image_size=image_data.image_size,
                        seed=1,
                        model_dir=save_dir,
                        timestamp=timestamp)

if role == 'master':
    from mindware.components.optimizers.amfes_optimizer import AsyncMFES

    method_name = 'mfeshb'
    problem_name = 'countingones'
    seed = 1
    optimizer = AsyncMFES(
        evaluator, config_space, 'hpo',
        eta=eta, random_state=seed, restart_needed=True,
        time_limit_per_trial=999999,
        runtime_limit=180, eval_type='partial', timestamp=timestamp,
        ip='', port=port, authkey=b'abc', output_dir=save_dir,
    )
    optimizer.run()
    # print('===== Optimization Finished =====')
    # print('> last 3 records:')
    # print(optimizer.recorder[-3:])
    # print('> incumbent configuration and performance:')
    # print(optimizer.get_incumbent())


else:
    worker = Worker(evaluator, ip, port, authkey=b'abc')
    worker.run()

# image_data.set_test_path(data_dir)
# print(clf.predict_proba(image_data))
# print(clf.predict(image_data))

# shutil.rmtree(save_dir)
