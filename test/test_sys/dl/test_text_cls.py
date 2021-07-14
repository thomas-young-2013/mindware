import os
import sys
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())
from mindware.components.models.text_classification.naivebert import NaiveBertClassifier
from mindware.components.models.text_classification.dpcnnbert import DPCNNBertClassifier
from mindware.components.models.text_classification.rcnnbert import RCNNBertClassifier
from mindware.estimators import TextClassifier
from mindware.datasets.text_dataset import TextDataset

mode = 'fit'

if mode == 'fit':
    dataset = TextDataset('data/text_datasets/ag_news_csv/train.csv',
                          config_path='./mindware/components/models/text_classification/nn_utils/bert-base-uncased')
    clf = TextClassifier(time_limit=10,
                         include_algorithms=['naivebert'],
                         ensemble_method='ensemble_selection')
    clf.fit(dataset)
    dataset.set_test_path('data/text_datasets/ag_news_csv/test.csv')
    print(clf.predict_proba(dataset))
    print(clf.predict(dataset))
else:
    config = NaiveBertClassifier.get_hyperparameter_search_space().sample_configuration().get_dictionary()
    config['epoch_num'] = 150
    config['device'] = 'cuda'
    clf = NaiveBertClassifier(**config)

    dataset = TextDataset('data/text_datasets/ag_news_csv/train.csv',
                          config_path='./mindware/components/models/text_classification/nn_utils/bert-base-uncased')
    dataset.load_data()
    dataset.set_test_path('data/text_datasets/ag_news_csv/test.csv')
    dataset.load_test_data()
    # train_dataset = TextBertDataset('data/text_datasets/ag_news_csv/train.csv',
    #                                 config_path='./mindware/components/models/text_classification/nn_utils/bert-base-uncased')
    # test_dataset = TextBertDataset('data/text_datasets/ag_news_csv/test.csv',
    #                                config_path='./mindware/components/models/text_classification/nn_utils/bert-base-uncased')

    clf.fit(dataset)
    print(clf.predict(dataset.test_dataset))
    print(clf.predict_proba(dataset.test_dataset))
    print(clf.score(dataset, accuracy_score))
    dataset.val_dataset = dataset.train_dataset
    print(clf.score(dataset, accuracy_score))
