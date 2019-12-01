import os
import pickle
import argparse
import numpy as np
from tabulate import tabulate

data_dir = './'
models_collection = ['lr', 'svm', 'dt', 'adb', 'rf', 'gb', 'knn']

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--mth', type=int, default=0)
parser.add_argument('--datasets', type=str, default='credit_g')
args = parser.parse_args()


def get_clf(model_type, seed=42):
    nj = 4
    if model_type == 'lr':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=seed)
    elif model_type == 'svm':
        from sklearn import svm
        clf = svm.SVC()
    elif model_type == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=seed)
    elif model_type == 'adb':
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=100, random_state=seed)
    elif model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=nj)
    elif model_type == 'gb':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(random_state=seed)
    elif model_type == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_jobs=nj)
    else:
        raise ValueError('Invalid MODEL!')
    return clf


def evaluate_models(model_type, data, train_phase=True, seed=42):
    from sklearn.model_selection import train_test_split
    if len(data) == 4:
        X_train, X_test, y_train, y_test = data
    elif len(data) == 2:
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    else:
        raise ValueError('Wrong data input!')

    clf = get_clf(model_type, seed)

    if train_phase is True:
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    else:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
    return score


def evaluate_joint_case(dataset, rep=5):
    save_path = data_dir + 'data/joint_eval_%s.pkl' % dataset
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            mean_result = pickle.load(f)
    else:
        results = dict()
        for model in models_collection:
            results[model] = list()

        np.random.seed(42)
        for rep_id in range(rep):
            print('='*20, 'Trial %d' % rep_id, '='*20)
            X, y, _ = load_data(dataset)
            seed = np.random.randint(10000000)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            for model in models_collection:
                tmp_res = list()

                from automlToolkit.fe_methods.evaluation_based_search import EvaluationBasedSearch
                fe = EvaluationBasedSearch(get_clf(model))
                X_train_new = fe.fit(X_train, y_train)
                X_test_new = fe.transform(X_test)

                for model_tmp in models_collection:
                    val_res = evaluate_models(model_tmp, (X_train_new, X_test_new, y_train, y_test))
                    test_res = evaluate_models(model_tmp, (X_train_new, X_test_new, y_train, y_test), train_phase=False)
                    tmp_res.append([val_res, test_res])
                results[model].append(tmp_res)

        mean_result = dict()
        for model in models_collection:
            tmp_res = np.array(results[model])
            mean_result[model] = np.mean(tmp_res, axis=0)

        with open(save_path, 'wb') as f:
            pickle.dump(mean_result, f)

    # Visualize the result.
    mode = args.mode
    idx = 0 if mode == 0 else 1
    print('='*20, 'Validation result' if mode == 0 else 'Test result')
    headers = ['eval_m']
    headers.extend(models_collection)
    data = list()
    for model_name in models_collection:
        row = [model_name]
        row.extend(mean_result[model_name][:, idx])
        data.append(row)
    print(tabulate(data, headers, tablefmt="github", floatfmt=".4f"))


def evaluate_selection_methods(dataset, rep=5):
    from automlToolkit.fe_methods.enumeration_selection import FeatureEnumerationSelector
    save_path = data_dir + 'data/enumeration_selection_eval_%s.pkl' % dataset
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            mean_result = pickle.load(f)
    else:
        results = dict()
        for model in models_collection:
            results[model] = list()

        np.random.seed(42)
        for rep_id in range(rep):
            print('=' * 20, 'Trial %d' % rep_id, '=' * 20)
            # X, y, _ = load_data(dataset)
            X, y, _ = None, None, None
            seed = np.random.randint(10000000)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            fe = FeatureEnumerationSelector()
            X_train_new = fe.fit(X_train, y_train)
            X_test_new = fe.transform(X_test)

            for model in models_collection:
                val_res = evaluate_models(model, (X_train_new, X_test_new, y_train, y_test))
                test_res = evaluate_models(model, (X_train_new, X_test_new, y_train, y_test), train_phase=False)
                results[model].append([val_res, test_res])

        mean_result = dict()
        for model in models_collection:
            tmp_res = np.array(results[model])
            mean_result[model] = np.mean(tmp_res, axis=0)

        with open(save_path, 'wb') as f:
            pickle.dump(mean_result, f)

    mode = args.mode
    idx = 0 if mode == 0 else 1
    print('='*20, 'Validation result' if mode == 0 else 'Test result')

    headers = models_collection
    data = [[mean_result[item][idx] for item in models_collection]]
    print(tabulate(data, headers, tablefmt="github", floatfmt=".4f"))


if __name__ == '__main__':
    if args.mth == 0:
        for dataset in args.datasets.split(','):
            evaluate_joint_case(dataset)
    if args.mth == 1:
        for dataset in args.datasets.split(','):
            evaluate_selection_methods(dataset)
