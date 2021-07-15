# Quick Start

This tutorial helps you run your first example with **MindWare**.

## Data Preparation

First, **prepare data** for the end-to-end AutoML system.
Here we use the iris dataset from sklearn as an example.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
```

After loading and splitting the dataset, wrap the data by **DataManager**.

```python
from mindware.utils.data_manager import DataManager

dm = DataManager(X_train, y_train)
train_data = dm.get_data_node(X_train, y_train)
test_data = dm.get_data_node(X_test, y_test)
```

## Optimization

**MindWare** provides an efficient way to complete machine learning task in an end-to-end manner.
In this example, we use <font color=#FF0000>**Classifier**</font> for the classification task.
Please specify **time_limit** to set the time budget for optimization.
Then simply call <font color=#FF0000>**Classifier.fit**</font> and the system will perform feature engineering, 
model selection, hyper-parameter optimization and model ensemble automatically. 
For large search spaces, the system employs search space decomposition to accelerate optimization.

```python
from mindware.estimators import Classifier

clf = Classifier(time_limit=3600)
clf.fit(train_data)
```

After optimization, call <font color=#FF0000>**Classifier.predict**</font> to get predictions of test dataset,
made by the best searched feature engineering method, ML model (might be ensemble model) and its hyper-parameters.

```python
pred = clf.predict(test_data)
```

(todo: result and visualization)
