import numpy as np
from sklearn.model_selection import cross_val_score
from operators.unary import unary_collection, op_dict


class EvaluationBasedSearch(object):
    def __init__(self, clf):
        self.transforms = list()
        self.classifier = clf
        self.val_fold = 5

    def fit(self, X, y):
        # Using Beam Search.
        feature_num = X.shape[1]
        tmp_features = X.copy()
        base_perf = cross_val_score(self.classifier, X, y, cv=self.val_fold).mean()

        for i in range(feature_num):
            print('Processing %d-th feature' % i)
            feature = X[:, i].copy()
            cache_features = None
            tmp_op = None
            for op_name in unary_collection:
                op = op_dict[op_name]
                new_feature = op.operate(list(feature))
                new_X = np.hstack((tmp_features, new_feature.reshape(-1, 1)))
                perf = cross_val_score(self.classifier, new_X, y, cv=self.val_fold).mean()
                if perf > base_perf:
                    cache_features = new_X
                    print('Perf improvement: %.4f => %.4f' % (base_perf, perf))
                    base_perf = perf
                    tmp_op = op_name
            if tmp_op is not None:
                self.transforms.append((i, tmp_op))
            if cache_features is not None:
                tmp_features = cache_features

        print(X.shape, '===>', tmp_features.shape)
        return tmp_features

    def transform(self, X):
        tmp_features = X.copy()
        for f_index, op_name in self.transforms:
            op = op_dict[op_name]
            new_feature = op.operate(list(X[:, f_index]))
            tmp_features = np.hstack((tmp_features, new_feature.reshape(-1, 1)))
        return tmp_features
