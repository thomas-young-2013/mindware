<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/mindware/master/docs/imgs/logo.png" width="40%">
</p>

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/thomas-young-2013/mindware/blob/master/LICENSE)
[![Build Status](https://api.travis-ci.org/thomas-young-2013/mindware.svg?branch=master)](https://www.travis-ci.org/github/thomas-young-2013/mindware)
[![Issues](https://img.shields.io/github/issues-raw/thomas-young-2013/mindware.svg)](https://github.com/thomas-young-2013/mindware/issues?q=is%3Aissue+is%3Aopen)
[![Bugs](https://img.shields.io/github/issues/thomas-young-2013/mindware/bug.svg)](https://github.com/thomas-young-2013/mindware/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![Pull Requests](https://img.shields.io/github/issues-pr-raw/thomas-young-2013/mindware.svg)](https://github.com/thomas-young-2013/mindware/pulls?q=is%3Apr+is%3Aopen)
[![Version](https://img.shields.io/github/release/thomas-young-2013/mindware.svg)](https://github.com/thomas-young-2013/mindware/releases) [![Join the chat at https://gitter.im/volcano-ml](https://badges.gitter.im/volcano-ml.svg)](https://gitter.im/volcano-ml?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Documentation Status](https://readthedocs.org/projects/mindware/badge/?version=latest)](https://mindware.readthedocs.io/en/latest/?badge=latest)

[MindWare Doc](https://mindware.readthedocs.io) | [MindWare中文文档](https://mindware.readthedocs.io/zh_CN/latest/)

## MindWare: Efficient Open-source AutoML System.
**MindWare** is an efficient open-source system to help users to automate the process of 1) data pre-processing, 2) feature engineering, 3) algorithm selection, 4) architecture design, 5) hyper-parameter tuning, and 6) model ensembling.
It is capable of improving its AutoML power by decomposing the entire large AutoML search space into small ones, and solve each sub-problems jointly and efficiently.
MindWare is designed and developed by the AutoML team from the <a href="http://net.pku.edu.cn/~cuibin/" target="_blank" rel="nofollow">DAIR Lab</a> at Peking University.
The goal is to make machine learning easier to apply both in industry and academia, and help facilitate data science.


## Who Should Consider MindWare
* Non-expert users who want to use machine learning techniques in their applications.
* ML Platform owners who want to support AutoML in their platform.
* Researchers and data scientists who want to experiment new AutoML algorithms, may it be: hyper-parameter tuning algorithm, neural architect search method, etc.


## Design Principles

- __User friendliness.__ MindWare needs few human assistance. To use MindWare, the users can define the task by writing only a few lines of code, regardless of the techinical details of the execution of the system.
- __High extensibility.__ New state-of-the-art ML algorithms or feature engineer operations can be added to the system. The decomposition techniques in MindWare ensures the efficiency of finding the best configurations over the enlarged search space. 
- __Advanced characteristic.__ MindWare provides special supports for large datasets. In addition, MindWare enables transfer-learning, meta-learning techniques to make AutoML with more intelligent behaviors.


## MindWare Capability in a Glance
<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/mindware/master/docs/imgs/mindware_framework.png" width="80%">
</p>


## Installation
MindWare requires *python>=3.6*. There are two ways to install MindWare:
### Installation via pip
MindWare is available on PyPI. You can install it by tying:

```sh
pip install mindware
```

### Manual installation
If you want to try the latest version, please manually install MindWare from source code by:

```sh
git clone https://github.com/thomas-young-2013/mindware.git && cd mindware
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install --user
```

For more detailed installation instructions, please refer to our [Installation Guide Document](https://mindware.readthedocs.io/en/latest/installation/installation_guide.html).

## Example

Here is a brief example that uses the package.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mindware.utils.data_manager import DataManager
from mindware.estimators import Classifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
dm = DataManager(X_train, y_train)
train_data = dm.get_data_node(X_train, y_train)
test_data = dm.get_data_node(X_test, y_test)

clf = Classifier(time_limit=3600)
clf.fit(train_data)

pred = clf.predict(test_data)
```

For more details and characteristics, please read the [documents](https://mindware.readthedocs.io/en/latest/?badge=latest) and [examples](https://github.com/thomas-young-2013/mindware/tree/master/examples/ci_examples/).


## **Releases and Contributing**
MindWare has a frequent release cycle. Please let us know if you encounter a bug by [filling an issue](https://github.com/thomas-young-2013/mindware/issues/new/choose).

We appreciate all contributions. If you are planning to contribute any bug-fixes, please do so without further discussions.

If you plan to contribute new features, new modules, etc. please first open an issue or reuse an existing issue, and discuss the feature with us.

To learn more about making a contribution to MindWare, please refer to our [How-to contribution page](https://github.com/thomas-young-2013/mindware/blob/master/CONTRIBUTING.md). 

We appreciate all contributions and thank all the contributors!



## **Feedback**
* Check [the existing open and closed issues](https://github.com/thomas-young-2013/mindware/issues?q=is%3Aissue).
* [File an issue](https://github.com/thomas-young-2013/mindware/issues/new/choose) on GitHub.
* Discuss on the MindWare [Gitter](https://gitter.im/volcano-ml?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge).



## **Related Projects**

Targeting at openness and advancing state-of-art technology, we have also released several open source projects.

* [OpenBOX](https://github.com/PKU-DAIR/open-box): an open source system and service to efficiently solve generalized blackbox optimization problems.

We encourage researchers to leverage the project to accelerate the AI development and research.


## **Related Publications**
**VolcanoML: Speeding up End-to-End AutoML via Scalable Search Space Decomposition**
Yang Li, Yu Shen, Wentao Zhang, Jiawei Jiang, Bolin Ding, Yaliang Li, Jingren Zhou, Zhi Yang, Wentao Wu, Ce Zhang and Bin Cui
International Conference on Very Large Data Bases (VLDB 2021).
https://arxiv.org/abs/2107.08861



**Efficient Automatic CASH via Rising Bandits**  
Yang Li, Jiawei Jiang, Jinyang Gao, Yingxia Shao, Ce Zhang and Bin Cui
Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2020). 
https://ojs.aaai.org/index.php/AAAI/article/view/5910



**MFES-HB: Efficient Hyperband with Multi-Fidelity Quality Measurements**
Yang Li, Yu Shen, Jiawei Jiang, Jinyang Gao, Ce Zhang and Bin Cui
Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2021). 
https://arxiv.org/abs/2012.03011



**OpenBox: A Generalized Black-box Optimization Service**
Yang Li, Yu Shen, Wentao Zhang, Yuanwei Chen, Huaijun Jiang, Mingchao Liu, Jiawei Jiang, Jinyang Gao, Wentao Wu, Zhi Yang, Ce Zhang and Bin Cui
ACM SIGKDD Conference on Knowledge Discovery and Data Mining (SIGKDD 2021).
https://arxiv.org/abs/2106.00421


## **License**

The entire codebase is under [MIT license](LICENSE).
