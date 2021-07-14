import os
import sys
import argparse
sys.path.append(os.getcwd())
from mindware.datasets.utils import load_data

parser = argparse.ArgumentParser()
dataset_set = 'diabetes,spectf,credit,ionosphere,lymphography,pc4,' \
              'messidor_features,winequality_red,winequality_white,splice,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default=dataset_set)
args = parser.parse_args()

for dataset in args.datasets.split(','):
    raw_data = load_data(dataset, datanode_returned=True)
    print(raw_data)
