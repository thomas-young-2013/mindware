import os
import sys
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())
from solnml.components.models.text_classification.naivebert import NaiveBertClassifier
from solnml.components.models.text_classification.dpcnnbert import DPCNNBertClassifier
from solnml.components.models.text_classification.rcnnbert import RCNNBertClassifier
from solnml.estimators import TextClassifier
from solnml.datasets.text_dataset import TextDataset

mode = 'fit'

if mode == 'fit':
    dataset = TextDataset('data/text_datasets/ag_news_csv/train.csv',
                          config_path='/Users/shenyu/PycharmProjects/automl-toolkit/solnml/components/models/text_classification/nn_utils/bert-base-uncased')
    clf = TextClassifier(time_limit=30,
                         include_algorithms=['rcnnbert'],
                         ensemble_method='ensemble_selection')
    clf.fit(dataset)
    dataset.load_test_data('data/text_datasets/ag_news_csv/test.csv')
    print(clf.predict_proba(dataset))
    print(clf.predict(dataset))
else:
    config = RCNNBertClassifier.get_hyperparameter_search_space().sample_configuration().get_dictionary()
    config['epoch_num'] = 1
    clf = RCNNBertClassifier(**config)
    dataset = TextDataset('data/text_datasets/ag_news_csv/train.csv',
                          config_path='/Users/shenyu/PycharmProjects/automl-toolkit/solnml/components/models/text_classification/nn_utils/bert-base-uncased')
    dataset.load_data()
    dataset.load_test_data('data/text_datasets/ag_news_csv/test.csv')
    # train_dataset = TextBertDataset('data/text_datasets/ag_news_csv/train.csv',
    #                                 config_path='./solnml/components/models/text_classification/nn_utils/bert-base-uncased')
    # test_dataset = TextBertDataset('data/text_datasets/ag_news_csv/test.csv',
    #                                config_path='./solnml/components/models/text_classification/nn_utils/bert-base-uncased')

    clf.fit(dataset)
    print(clf.predict(dataset.test_dataset))
    print(clf.predict_proba(dataset.test_dataset))
    print(clf.score(dataset, accuracy_score))
    dataset.val_dataset = dataset.train_dataset
    print(clf.score(dataset, accuracy_score))
