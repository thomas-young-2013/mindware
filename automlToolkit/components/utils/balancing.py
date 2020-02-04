import numpy as np


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
