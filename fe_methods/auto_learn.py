import numpy as np

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SelectKBest
from scipy.spatial.distance import pdist, squareform


class AutoLearn:
    """
    Reference: AutoLearn - Automated Feature Generation and Selection, ICDM 2017
    """
    def __init__(self, eta_1, eta_2):
        self.eta_1 = eta_1
        self.eta_2 = eta_2

    def dcor(self, f1, f2):
        f1 = np.atleast_1d(f1)
        f2 = np.atleast_1d(f2)
        if np.prod(f1.shape) == len(f1):
            f1 = f1[:, None]
        if np.prod(f2.shape) == len(f2):
            f2 = f2[:, None]
        f1 = np.atleast_2d(f1)
        f2 = np.atleast_2d(f2)

        n = f1.shape[0]
        if f2.shape[0] != f1.shape[0]:
            raise ValueError("Number of samples must match")

        a = squareform(pdist(f1))
        b = squareform(pdist(f2))

        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        res = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

        return res

    def preprocessing(self, x_train, x_test, labels):
        print("start preprocessing......")
        assert len(x_train.shape) == 2 and len(x_test.shape) == 2

        ig_scores = mutual_info_classif(x_train, labels)

        selected_indices = ig_scores > self.eta_1
        print(str(sum(selected_indices)), "features are selected after preprocessed")
        return x_train[:, selected_indices], x_test[:, selected_indices]

    @staticmethod
    def feature_selection(constructed_features, labels):
        feature_num = constructed_features.shape[1]
        selector = SelectKBest(mutual_info_classif, k=int(np.sqrt(feature_num)))
        selector.fit(constructed_features, labels)
        return selector.transform(constructed_features)

    @staticmethod
    def generate_feature(fi, fj, is_kernel):
        if is_kernel:
            lr = KernelRidge()
        else:
            lr = Ridge()

        lr.fit(fi.reshape(-1, 1), fj)
        f_pred = lr.predict(fi.reshape(-1, 1))

        return f_pred, fj - f_pred

    def fit_transform(self, x_train, x_test, y_train):
        assert x_train.shape[1] == x_test.shape[1]

        # preprocessing
        x_train, x_test = self.preprocessing(x_train, x_test, y_train)
        x = np.vstack((x_train, x_test))
        feature_num = x_train.shape[1]
        train_num = x_train.shape[0]

        constructed_features = []
        if feature_num == 0:
            raise ValueError("the size of original space is zero!!!")

        for i in range(feature_num):
            for j in range(feature_num):
                dcor_val = self.dcor(x[:, i], x[:, j])
                if i != j and dcor_val != 0:
                    feature_pair = None
                    if 0 < dcor_val < self.eta_2:
                        feature_pair = self.generate_feature(x[:, i], x[:, j], True)

                    if self.eta_2 <= dcor_val <= 1:
                        feature_pair = self.generate_feature(x[:, i], x[:, j], False)

                    constructed_features.append(feature_pair[0])
                    constructed_features.append(feature_pair[1])

        constructed_features = np.array(constructed_features).T
        constructed_features = self.feature_selection(constructed_features, y_train)
        return constructed_features[: train_num], constructed_features[train_num: len(constructed_features)]
