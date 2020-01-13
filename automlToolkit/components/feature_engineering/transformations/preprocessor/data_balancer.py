import numpy as np
from collections import Counter
from automlToolkit.components.feature_engineering.transformations.base_transformer import *


class DataBalancer(Transformer):
    def __init__(self, threshold=0.6, random_state=1):
        super().__init__("data_balancer", 20)
        self.threshold = threshold
        self.random_state = random_state

    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data
        if y is None:
            data = (X.copy(), None)
        else:
            label_idx_dict = {}
            for i, label in enumerate(y):
                if label not in label_idx_dict:
                    label_idx_dict[label] = [i]
                else:
                    label_idx_dict[label].append(i)

            counts = list(Counter(y.copy()).values())
            median = np.median(counts)
            min_cnt, max_cnt = np.min(counts), np.max(counts)

            if min_cnt >= self.threshold * median:
                data = [X.copy(), y.copy]
            else:
                np.random.seed(self.random_state)
                resample_num = int(median * self.threshold)
                copy_X, copy_y = X.copy(), y.copy()
                print('Before balancing', Counter(y.copy()))
                for key in label_idx_dict:
                    length = len(label_idx_dict[key])
                    if length < resample_num:
                        copy = int(resample_num / length)
                        left = resample_num - copy * length
                        copy -= 1
                        for _ in range(copy):
                            copy_X = np.vstack((copy_X, copy_X[label_idx_dict[key]].copy()))
                            copy_y = np.hstack((copy_y, copy_y[label_idx_dict[key]].copy()))
                        left_idx_list = np.random.choice(label_idx_dict[key], left, replace=False)
                        copy_X = np.vstack((copy_X, copy_X[left_idx_list].copy()))
                        copy_y = np.hstack((copy_y, copy_y[left_idx_list].copy()))
                data = (copy_X, copy_y)
                print('After balancing', Counter(copy_y.copy()))
        new_feature_types = input_datanode.feature_types.copy()
        output_datanode = DataNode(data, new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)

        return output_datanode
