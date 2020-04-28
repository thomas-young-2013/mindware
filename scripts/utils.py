import os
import re
data_folder = 'data/meta_res_cp/'
filename_list = os.listdir(data_folder)
algorithms = ['lightgbm', 'random_forest',
              'libsvm_svc', 'extra_trees',
              'liblinear_svc', 'k_nearest_neighbors',
              'logistic_regression',
              'gradient_boosting', 'adaboost']
algo_str = '|'.join(algorithms)
pattern = '(.*)_(%s)_(\d+)_(\d+)_20.pkl' % algo_str
print(pattern)
for _filename in filename_list:
    used_name = data_folder + _filename
    if _filename.startswith('.'):
        continue

    print(_filename)
    result = re.search(pattern, _filename, re.I)
    dataset, algo, run_id, seed = result.group(1), result.group(2), result.group(3), result.group(4)
    new_name = data_folder + "%s-%s-%s-%d-%d.pkl" % (dataset, algo, 'acc', int(run_id), 20)
    os.rename(used_name, new_name)
    print("used_name: %s,new_name: %s" % (used_name, new_name))
