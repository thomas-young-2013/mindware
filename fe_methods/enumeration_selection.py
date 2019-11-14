import numpy as np
from operators.unary import unary_collection, op_dict


class FeatureEnumerationSelector(object):
    def __init__(self):
        self.selector = None

    def generate(self, X):
        features = X.copy()
        # Generate features in the feature space.
        for i in range(X.shape[1]):
            print('Processing %d-th feature' % i)
            feature = X[:, i].copy()
            tmp_features = None
            for op_name in unary_collection:
                op = op_dict[op_name]
                new_feature = np.array(op.operate(list(feature)))
                if tmp_features is not None:
                    tmp_features = np.hstack((tmp_features, new_feature.reshape(-1, 1)))
                else:
                    tmp_features = new_feature.reshape(-1, 1)
            features = np.hstack((features, tmp_features))
        return features

    def fit(self, X, y):
        feature_num = X.shape[1]
        features = self.generate(X)

        # Select a subset of candidate features.
        from sklearn.feature_selection import SelectKBest, f_classif
        k_num = feature_num * 2
        self.selector = SelectKBest(f_classif, k=k_num)
        self.selector.fit(features, y)
        X_new = self.selector.transform(features)
        print(X.shape, '===>', X_new.shape)
        return X_new

    def transform(self, X_test):
        features = self.generate(X_test)
        return self.selector.transform(features)
