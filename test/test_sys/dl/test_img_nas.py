import os
import sys

sys.path.append(os.getcwd())

from mindware.datasets.image_dataset import ImageDataset
from mindware.estimators import ImageClassifier

data_dir = 'data/img_datasets/extremely_small/'
# data_dir = 'data/img_datasets/cifar10/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True)
save_dir = './data/eval_exps/mindware'
clf = ImageClassifier(time_limit=300,
                      max_epoch=8,
                      mode='search',
                      space='nasbench201',
                      ensemble_method='ensemble_selection',
                      # ensemble_method=None,
                      evaluation='holdout',
                      skip_profile=True,
                      output_dir=save_dir,
                      n_jobs=1)
clf.fit(image_data, opt_method='whatever')
image_data.set_test_path(data_dir)
print(clf.predict_proba(image_data))
print(clf.predict(image_data))

# shutil.rmtree(save_dir)
