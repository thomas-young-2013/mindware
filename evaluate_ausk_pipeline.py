import numpy as np
import random
from components.evaluator import Evaluator
from evaluate_transgraph import engineer_data
from utils.default_random_forest import DefaultRandomForest


def eval(dataset):
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    data_node, fe_time = engineer_data(dataset, fe="none", time_budget=10000, seed=seed)

    cs = DefaultRandomForest.get_hyperparameter_search_space()
    config = cs.get_default_configuration().get_dictionary()
    clf = DefaultRandomForest(**config)
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(class_weight='balanced')
    evaluator = Evaluator(seed=seed, clf=clf)

    X, y = data_node.data
    from sklearn.preprocessing import StandardScaler, Imputer
    X = StandardScaler().fit_transform(X)
    X = Imputer(strategy='most_frequent').fit_transform(X)

    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_resample(X, y)
    # X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    data = [X, y]
    data_node.data = data
    score = evaluator(data_node)
    print(score)


if __name__ == "__main__":
    eval('credit')
