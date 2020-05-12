from solnml.datasets.utils import load_train_test_data
from solnml.estimators import Classifier


def evaluate_package():
    train_data, test_data = load_train_test_data('pc4', data_dir='./')
    Classifier().fit(train_data)


if __name__ == "__main__":
    evaluate_package()
