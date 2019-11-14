import sys

sys.path.append('/home/daim_gpu/sy/AlphaML')
sys.path.append('/home/daim_gpu/sy/Feature-Engineering')

from operators.unary import *
from lfe import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--module', choices=['gen', 'save', 'load', 'fit', 'predict'], default='save')
parser.add_argument('--dataset_name', type=str, default='pc4')
args = parser.parse_args()


def test():
    module = args.module
    dataset_name = args.dataset_name
    if module == 'save':
        test_save(dataset_name)
    elif module == 'gen':
        test_generate(dataset_name)
    elif module == 'load':
        test_load()
    elif module == 'fit':
        test_fit()
    elif module == 'predict':
        test_predict(dataset_name)


def test_save(dataset_name='pc4'):
    import os
    import pickle as pkl
    with open(os.path.join('lfe/data', 'qsa_' + dataset_name), 'rb') as f:
        qsa = pkl.load(f)
    with open(os.path.join('lfe/data', 'label_' + dataset_name), 'rb') as f:
        label = pkl.load(f)
        from collections import Counter
        for i in label:
            print(i, Counter(label[i]))
    # print(qsa, label)


def test_operator():
    col = [-1, 1, 2, 3, -4, 4, 4, 5, 6, 100]
    logop = LogOperator()
    sqrtop = SqrtOperator()
    sqop = SquareOperator()
    fop = FreqOperator()
    rop = RoundOperator()
    tanhop = TanhOperator()
    sigop = SigmoidOperator()
    irop = IsotonicOperator()
    zsop = ZscoreOperator()
    nmop = NormalizeOperator()
    print(logop.operate(col))
    print(sqrtop.operate(col))
    print(sqop.operate(col))
    print(fop.operate(col))
    print(rop.operate(col))
    print(tanhop.operate(col))
    print(sigop.operate(col))
    print(irop.operate(col))
    print(zsop.operate(col))
    print(nmop.operate(col))


def test_valid():
    from alphaml.datasets.cls_dataset.dataset_loader import load_data

    lfe = LFE()
    x, y, _ = load_data('pc4')
    x = np.array(x, dtype=float)
    print(valid_sample(x, y, 0))


def test_generate(dataset_name='pc4'):
    from alphaml.datasets.cls_dataset.dataset_loader import load_data
    lfe = LFE()
    x, y, _ = load_data(dataset_name)
    x = np.array(x, dtype=float)
    qsa, dict = lfe.generate_samples(x, y, dataset_name)
    print(qsa, dict)
    print(qsa.shape)


def test_lfe(dataset_name='pc4'):
    from alphaml.datasets.cls_dataset.dataset_loader import load_data
    lfe = LFE()
    x, y, _ = load_data(dataset_name)
    x = np.array(x, dtype=float)
    qsa, dict = lfe.generate_samples(x, y, dataset_name)
    print(qsa, dict)
    print(qsa.shape)


def test_load(data_dir='lfe/data'):
    lfe = LFE()
    x, y = lfe.load_training_data(data_dir)
    print(x)
    from collections import Counter
    for key in y:
        print(Counter(y[key]))


def test_fit(data_dir='lfe/data', save_dir='lfe'):
    lfe = LFE()
    train_ops = ['norm']
    lfe.fit(train_ops, data_dir=data_dir, save_dir=save_dir)


def test_predict(dataset_name='pc4', save_dir='lfe'):
    from alphaml.datasets.cls_dataset.dataset_loader import load_data
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    lfe = LFE()
    x, y, _ = load_data(dataset_name)
    x = np.array(x, dtype=float)
    result_ori = 0
    result_new = 0
    for i in range(10):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y)

        for i in range(5):
            RF = RandomForestClassifier()
            RF.fit(train_x, train_y)
            pred_ori = RF.predict(test_x)
            result_ori += f1_score(pred_ori, test_y) / 50

        tran = lfe.choose(train_x, train_y, save_dir)
        num_features = train_x.shape[1]
        for i in range(num_features):
            if tran[i] is not None:
                col = op_dict[tran[i]].operate(train_x[:, i])
                col = np.reshape(col, (len(col), 1))
                train_x = np.hstack((train_x, col))

                col = op_dict[tran[i]].operate(test_x[:, i])
                col = np.reshape(col, (len(col), 1))
                test_x = np.hstack((test_x, col))

        for i in range(5):
            RF = RandomForestClassifier()
            RF.fit(train_x, train_y)
            pred_new = RF.predict(test_x)
            result_new += f1_score(pred_new, test_y) / 50

    print("Ori:", result_ori)
    print("New:", result_new)


test()
