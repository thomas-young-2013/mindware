![](docs/logos/soln_ml_300.jpg)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/thomas-young-2013/automl-toolkit/blob/master/LICENSE)

## Soln-ML: Towards Self-Learning AutoML System.
Soln-ML is an AutoML system, which is capable of improving its AutoML power by learning from past experience.
It implements many basic components that enables automatic machine learning. 
Furthermore, this toolkit can be also used to nourish new AutoML algorithms.
Soln-ML is developed by <a href="http://net.pku.edu.cn/~cuibin/" target="_blank" rel="nofollow">DAIM Lab</a> at Peking University.
The goal of Soln-ML is to make machine learning easier to apply both in industry and academia.

## Guiding principles

- __User friendliness.__ Soln-ML needs few human assistance.

- __Easy extensibility.__ New ML algorithms are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making it suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

## Characteristics
- Soln-ML supports AutoML on large datasets.

- Soln-ML enables transfer-learning, meta-learning and reinforcement learning techniques to make AutoML with more intelligent behaviors.

## Example

Here is a brief example that uses the package.

```
from automlToolkit.estimators import Classifier

train_data = dm.get_data_node(X_train, y_train)
test_data = dm.get_data_node(X_test, y_test)
clf = Classifier(time_limit=time_limit)
clf.fit(train_data)
predictions = clf.predict(test_data)
```

For more details, please check [examples](https://github.com/thomas-young-2013/automl-toolkit/tree/master/examples).

## Installation

To install the package, please use the `pip` installation as follows:

```shell
pip3 install autokeras
```

Please follow the [installation guide](https://autokeras.com/install) for more details.

**Note:** Currently, Soln-ML is only compatible with **Python >= 3.5** and **Scikit-learn == 0.21.3**.
