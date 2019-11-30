import pandas as pd
from sklearn import preprocessing


def trans_label(input):
    le = preprocessing.LabelEncoder()
    le.fit(input)
    return le.transform(input)


def load_data(data_file):
    delimiter = ';'
    data = pd.read_csv(data_file, delimiter=delimiter).values
    return data[:, :-1], trans_label(data[:, -1])


dataset = 'winequality_white'
data_file = 'data/datasets/%s.csv' % dataset
X, y = load_data(data_file)
print(X.shape, y.shape, set(y))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
# X = MinMaxScaler().fit_transform(X)
# X = Imputer(strategy='mean').fit_transform(X)

from components.evaluator import cross_validation

score = cross_validation(clf, X, y, 5)
print(score)


"""
[(1.000000, SimpleClassificationPipeline(
{
'categorical_encoding:__choice__': 'no_encoding', 
'preprocessor:__choice__': 'no_preprocessing', 
'imputation:strategy': 'mean', 
'rescaling:__choice__': 'minmax', 
'balancing:strategy': 'none', 
'classifier:DefaultRandomForest:min_weight_fraction_leaf': 0.0, 
'classifier:DefaultRandomForest:min_impurity_decrease': 0.0, 
'classifier:DefaultRandomForest:min_samples_leaf': 1, 
'classifier:DefaultRandomForest:criterion': 'gini', 
'classifier:__choice__': 'DefaultRandomForest', 
'classifier:DefaultRandomForest:max_leaf_nodes': 'None', 
'classifier:DefaultRandomForest:max_features': 'auto', 
'classifier:DefaultRandomForest:max_depth': 'None', 
'classifier:DefaultRandomForest:n_estimators': 100, 
'classifier:DefaultRandomForest:bootstrap': 'True', 
'classifier:DefaultRandomForest:min_samples_split': 2},
dataset_properties={
  'target_type': 'classification',
  'multiclass': False,
  'task': 1,
  'sparse': False,
  'signed': False,
  'multilabel': False})),
]
"""

"""
SimpleClassificationPipeline({
'classifier:DefaultRandomForest:max_depth': 'None', 
'classifier:DefaultRandomForest:max_features': 'auto', 
'categorical_encoding:one_hot_encoding:use_minimum_fraction': 'True', 
'classifier:DefaultRandomForest:min_weight_fraction_leaf': 0.0, 
'rescaling:__choice__': 'minmax', 
'balancing:strategy': 'none', 
'classifier:DefaultRandomForest:min_samples_leaf': 1, 
'classifier:DefaultRandomForest:criterion': 'gini', 
'preprocessor:__choice__': 'kitchen_sinks', 
'preprocessor:kitchen_sinks:n_components': 808, 
'classifier:DefaultRandomForest:bootstrap': 'True', 
'preprocessor:kitchen_sinks:gamma': 3.2685920280477223, 
'classifier:DefaultRandomForest:max_leaf_nodes': 'None', 
'classifier:__choice__': 'DefaultRandomForest', 
'categorical_encoding:__choice__': 'one_hot_encoding', 
'classifier:DefaultRandomForest:min_impurity_decrease': 0.0, 
'classifier:DefaultRandomForest:n_estimators': 100, 
'classifier:DefaultRandomForest:min_samples_split': 2, 
'imputation:strategy': 'most_frequent', 
'categorical_encoding:one_hot_encoding:minimum_fraction': 0.061991393301031046},
"""

"""
white.
[(1.000000, SimpleClassificationPipeline({
'classifier:DefaultRandomForest:max_depth': 'None', 
'classifier:DefaultRandomForest:max_features': 'auto', 
'rescaling:quantile_transformer:n_quantiles': 1393, 
'categorical_encoding:one_hot_encoding:use_minimum_fraction': 'False', 
'classifier:DefaultRandomForest:min_weight_fraction_leaf': 0.0, 
'rescaling:__choice__': 'quantile_transformer', 
'balancing:strategy': 'weighting', 
'classifier:DefaultRandomForest:min_samples_leaf': 1, 
'classifier:DefaultRandomForest:criterion': 'gini', 
'preprocessor:__choice__': 'kitchen_sinks', 
'preprocessor:kitchen_sinks:n_components': 755, 
'classifier:DefaultRandomForest:bootstrap': 'True', 
'preprocessor:kitchen_sinks:gamma': 0.00018592906294321256, 
'classifier:DefaultRandomForest:max_leaf_nodes': 'None', 
'classifier:__choice__': 'DefaultRandomForest', 
'categorical_encoding:__choice__': 'one_hot_encoding', 
'rescaling:quantile_transformer:output_distribution': 'normal', 
'classifier:DefaultRandomForest:min_impurity_decrease': 0.0, 
'classifier:DefaultRandomForest:n_estimators': 100, 
'classifier:DefaultRandomForest:min_samples_split': 2, 
'imputation:strategy': 'median'}  
"""
