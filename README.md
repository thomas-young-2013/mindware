<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/soln-ml/dev_refactor/docs/logos/logo.jpg" width="68%">
</p>

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/thomas-young-2013/soln-ml/blob/master/LICENSE)
[![Build Status](https://api.travis-ci.org/thomas-young-2013/soln-ml.svg?branch=dev_refactor)](https://www.travis-ci.org/github/thomas-young-2013/soln-ml)
[![Issues](https://img.shields.io/github/issues-raw/thomas-young-2013/soln-ml.svg)](https://github.com/thomas-young-2013/soln-ml/issues?q=is%3Aissue+is%3Aopen)
[![Bugs](https://img.shields.io/github/issues/thomas-young-2013/soln-ml/bug.svg)](https://github.com/thomas-young-2013/soln-ml/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![Pull Requests](https://img.shields.io/github/issues-pr-raw/thomas-young-2013/soln-ml.svg)](https://github.com/thomas-young-2013/soln-ml/pulls?q=is%3Apr+is%3Aopen)
[![Version](https://img.shields.io/github/release/thomas-young-2013/soln-ml.svg)](https://github.com/thomas-young-2013/soln-ml/releases) [![Join the chat at https://gitter.im/volcano-ml](https://badges.gitter.im/volcano-ml.svg)](https://gitter.im/volcano-ml?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Documentation Status](https://readthedocs.org/projects/soln-ml/badge/?version=latest)](https://soln-ml.readthedocs.io/en/latest/?badge=latest)

------------------

## VolcanoML: Speeding up End-to-End AutoML via Scalable Search Space Decomposition.
Volcano-ML is a powerful AutoML system, which automates feature engineering, algorithm selection and hyperparameter tuning. 
It is capable of improving its AutoML power by decomposing the entire large AutoML search space into small ones.
The system executes like the eruption of a volcano, hence the name 'Volcano-ML'.

Volcano-ML is developed by <a href="http://net.pku.edu.cn/~cuibin/" target="_blank" rel="nofollow">DAIM Lab</a> at Peking University.
The goal of Volcano-ML is to make machine learning easier to apply both in industry and academia.
Currently, Volcano-ML is compatible with: **Python >= 3.6**.

------------------

## Characteristics

- __User friendliness.__ Volcano-ML needs few human assistance. To use Volcano-ML, the users can define the task by writing only a few lines of code, regardless of the techinical details of the execution of the system.
- __High extensibility.__ New state-of-the-art ML algorithms or feature engineer operations can be added to the system simply. The decomposition techniques in Volcano-ML ensures the efficiency of finding the best configurations over the enlarged search space. 
- __Advanced characteristic.__ Volcano-ML provides special supports for large datasets. In addition, Volcano-ML enables transfer-learning, meta-learning techniques to make AutoML with more intelligent behaviors.

------------------

## Releases
* New release: [v1.3]() -released on xx-xx-2021.

------------------

## Example

Here is a brief example that uses the package.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from solnml.utils.data_manager import DataManager
from solnml.estimators import Classifier

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

For more details and characteristics, please check [examples](https://github.com/thomas-young-2013/soln-ml/tree/master/examples/ci_examples/).

------------------
## Visualization
TODO.

------------------
## Installation

Before installing Volcano-ML, please install the necessary library [swig](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/).

Volcano-ML requires SWIG (>= 3.0, <4.0) as a build dependency, and we suggest you to download & install [swig=3.0.12](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/).


Then, you can install Volcano-ML itself. Volcano-ML supports and is tested on Ubuntu >= 16.04, macOS >= 10.14.1, and Windows 10 >= 1809. The installation requires a python environment that has `python 64-bit >= 3.6`.There are two ways to install Volcano-ML:

#### Installation via pip
Volcano-ML is available on PyPI. You can install it by tying:

```sh
pip install soln-ml
```

#### Manual installation from the github source

If you want to try latest code, please manually install Volcano-ML from source code by:

```sh
git clone https://github.com/thomas-young-2013/soln-ml.git && cd soln-ml
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

### Tips on Installing Swig

#### Linux:

On Arch Linux (or any distribution with swig4 as default implementation), you need to confirm that the version of SWIG is in (>= 3.0, <4.0).

We suggest you to install [swig=3.0.12](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/)..

```sh
./configure
make & make install
```

#### MACOSX:

Before installing SWIG, you need to install [pcre](https://sourceforge.net/projects/pcre/files/pcre/8.44/):

```sh
cd $pcre_dir
./configure
make & make install
```

Then add library path of `/usr/local/lib` for `pcre`:

```sh
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib
export LD_LIBRARY_PATH
```

Finally, install Swig:

```sh
cd $swig_dir
./configure
make & make install
```

Before installing python package `pyrfr=0.8.0`, download source code from [pypi](https://pypi.org/project/pyrfr/#files):

```sh
cd $pyrfr_dir
python setup.py install
```

#### Windows:

You need to download [swigwin](https://sourceforge.net/projects/swig/files/swigwin/swigwin-3.0.12/), and then install Soln-ML.

------------------
## **Feedback**
* Check [the existing open and closed issues](https://github.com/thomas-young-2013/soln-ml/issues?q=is%3Aissue).
* [File an issue](https://github.com/thomas-young-2013/soln-ml/issues/new/choose) on GitHub.
* Discuss on the Volcano-ML [Gitter](https://gitter.im/volcano-ml?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge).

------------------
## **Related Projects**

Targeting at openness and advancing state-of-art technology, we have also released another open source project.

* [OpenBOX](https://github.com/thomas-young-2013/open-box): an open source system and service to efficiently solve generalized blackbox optimization problems.

We encourage researchers to leverage the project to accelerate the AI development and research.

---------------------
## **Related Publications**
**VolcanoML: Speeding up End-to-End AutoML via Scalable Search Space Decomposition**
Yang Li, Yu Shen, Wentao Zhang, Jiawei Jiang, Bolin Ding, Yaliang Li, Jingren Zhou, Zhi Yang, Wentao Wu, Ce Zhang and Bin Cui
International Conference on Very Large Data Bases (VLDB 2021).



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


---------------------
## **License**

The entire codebase is under [MIT license](LICENSE).
