import openml
import numpy as np


def load_task(task_id):
    """Function used for loading data."""
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    print(X.shape, y.shape)
    train_indices, test_indices = task.get_train_test_split_indices()
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    print(X_train.shape, X_test.shape)
    cat = None
    return X_train, y_train, X_test, y_test, cat


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, cat = load_task(31)
    print(X_train.shape, X_test.shape)
    print(cat)
