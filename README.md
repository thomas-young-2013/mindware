![](docs/logos/soln_ml_300.jpg)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/thomas-young-2013/soln-ml/blob/master/LICENSE)
[![Build Status](https://api.travis-ci.org/thomas-young-2013/soln-ml.svg?branch=dev_refactor)](https://www.travis-ci.org/github/thomas-young-2013/soln-ml)
[![Issues](https://img.shields.io/github/issues-raw/thomas-young-2013/soln-ml.svg)](https://github.com/thomas-young-2013/soln-ml/issues?q=is%3Aissue+is%3Aopen)
[![Bugs](https://img.shields.io/github/issues/thomas-young-2013/soln-ml/bug.svg)](https://github.com/thomas-young-2013/soln-ml/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![Pull Requests](https://img.shields.io/github/issues-pr-raw/thomas-young-2013/soln-ml.svg)](https://github.com/thomas-young-2013/lite-bo/pulls?q=is%3Apr+is%3Aopen)
[![Version](https://img.shields.io/github/release/thomas-young-2013/soln-ml.svg)](https://github.com/thomas-young-2013/soln-ml/releases)
<!--[![Join the chat at https://gitter.im/bbo-open-box](https://badges.gitter.im/bbo-open-box.svg)](https://gitter.im/bbo-open-box?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)-->
[![Documentation Status](https://readthedocs.org/projects/soln-ml/badge/?version=latest)](https://soln-ml.readthedocs.io/en/latest/?badge=latest)

------------------

## Soln-ML: Towards Self-Improving AutoML System.
Soln-ML is an AutoML system, which is capable of improving its AutoML power by learning from past experience.
It implements many basic components that enables automatic machine learning. 
Furthermore, this toolkit can be also used to nourish new AutoML algorithms.
Soln-ML is developed by <a href="http://net.pku.edu.cn/~cuibin/" target="_blank" rel="nofollow">DAIM Lab</a> at Peking University.
The goal of Soln-ML is to make machine learning easier to apply both in industry and academia.

Currently, Soln-ML is compatible with: **Python >= 3.5**.

------------------

## Guiding principles

- __User friendliness.__ Soln-ML needs few human assistance.

- __Easy extensibility.__ New ML algorithms are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making it suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

------------------

## Characteristics
- Soln-ML supports AutoML on large datasets.

- Soln-ML enables transfer-learning, meta-learning and reinforcement learning techniques to make AutoML with more intelligent behaviors.

------------------

## Example

Here is a brief example that uses the package.

```python
from solnml.estimators import Classifier
clf = Classifier(dataset_name='iris',
                 time_limit=150,
                 output_dir='logs/',
                 ensemble_method='stacking',
                 evaluation='holdout',
                 metric='acc')
clf.fit(train_data)
predictions = clf.predict(test_data)
```

For more details, please check [examples](https://github.com/thomas-young-2013/soln-ml/tree/master/examples).

------------------

## Installation

Before installing Soln-ML, please install the necessary library [swig](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/).

Soln-ML requires SWIG (>= 3.0, <4.0) as a build dependency, and we suggest you to download & install [swig=3.0.12](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/).


Then, you can install Soln-ML itself. There are two ways to install Soln-ML:

#### Installation via pip
Soln-ML is available on PyPI. You can install it by tying:

```sh
pip install soln-ml
```

#### Manual installation from the github source

```sh
git clone https://github.com/thomas-young-2013/soln-ml.git && cd soln-ml
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

### Tips on Installing Swig


- **for Arch Linux User:**

On Arch Linux (or any distribution with swig4 as default implementation), you need to confirm that the version of SWIG is in (>= 3.0, <4.0).

We suggest you to install [swig=3.0.12](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/)..

```sh
./configure
make & make install
```

- **for MACOSX User:**

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

- **for Windows User:**

You need to download [swigwin](https://sourceforge.net/projects/swig/files/swigwin/swigwin-3.0.12/), and then install Soln-ML.
