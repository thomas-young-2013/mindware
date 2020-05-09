import numpy as np
from collections import Counter


def get_weights(Y, classifier, preprocessor, init_params, fit_params):
    if init_params is None:
        init_params = {}

    if fit_params is None:
        fit_params = {}

    # Classifiers which require sample weights:
    # We can have adaboost in here, because in the fit method,
    # the sample weights are normalized:
    # https://github.com/scikit-learn/scikit-learn/blob/0.15.X/sklearn/ensemble/weight_boosting.py#L121
    # Have RF and ET in here because they emit a warning if class_weights
    #  are used together with warmstarts
    clf_ = ['adaboost', 'random_forest', 'extra_trees', 'sgd', 'passive_aggressive']
    pre_ = []
    if classifier in clf_ or preprocessor in pre_:
        if len(Y.shape) > 1:
            offsets = [2 ** i for i in range(Y.shape[1])]
            Y_ = np.sum(Y * offsets, axis=1)
        else:
            Y_ = Y

        unique, counts = np.unique(Y_, return_counts=True)
        # This will result in an average weight of 1!
        cw = 1 / (counts / np.sum(counts)) / 2
        if len(Y.shape) == 2:
            cw /= Y.shape[1]

        sample_weights = np.ones(Y_.shape)

        for i, ue in enumerate(unique):
            mask = Y_ == ue
            sample_weights[mask] *= cw[i]

        if classifier in clf_:
            fit_params['sample_weight'] = sample_weights
        if preprocessor in pre_:
            fit_params['sample_weight'] = sample_weights

    # Classifiers which can adjust sample weights themselves via the
    # argument `class_weight`
    clf_ = ['decision_tree', 'liblinear_svc',
            'libsvm_svc']
    pre_ = ['liblinear_svc_preprocessor',
            'extra_trees_preproc_for_classification']
    if classifier in clf_:
        init_params['class_weight'] = 'balanced'
    if preprocessor in pre_:
        init_params['class_weight'] = 'balanced'

    clf_ = ['ridge']
    if classifier in clf_:
        class_weights = {}

        unique, counts = np.unique(Y, return_counts=True)
        cw = 1. / counts
        cw = cw / np.mean(cw)

        for i, ue in enumerate(unique):
            class_weights[ue] = cw[i]

        if classifier in clf_:
            init_params['class_weight'] = class_weights

    return init_params, fit_params


def get_data(X, y, threshold=0.6, random_state=1):
    if y is None:
        return X.copy(), None
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

        if min_cnt >= threshold * median:
            return X.copy(), y.copy()
        else:
            np.random.seed(random_state)
            resample_num = int(median * threshold)
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

            print('After balancing', Counter(copy_y.copy()))

            return copy_X, copy_y


def smote(X, y, random_state=1):
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE

    labels = list(y)
    cnts = list()
    for val in set(labels):
        cnts.append(labels.count(val))
    cnts = sorted(cnts)
    if cnts[0] < 6:
        sm = SMOTE(random_state=random_state, k_neighbors=cnts[0] - 1)
        model = SMOTEENN(random_state=random_state, smote=sm)
    else:
        # The default value of k_neighbors in SMOTEENN is 5
        model = SMOTEENN(random_state=random_state)

    X_res, y_res = model.fit_resample(X, y)
    return X_res, y_res
