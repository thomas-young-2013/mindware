# Installation Guide

## 1 System Requirements

Installation Requirements:
+ Python >= 3.6 (3.7 is recommended!)

Supported Systems:
+ Linux (Ubuntu, ...)
+ macOS
+ Windows

## 2 Preparations before Installation

We **STRONGLY** suggest you to create a Python environment via [Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
conda create -n mindware python=3.7
conda activate mindware
```

Then we recommend you to update your `pip` and `setuptools` as follows:
```bash
pip install pip setuptools --upgrade
```

## 3 Install MindWare

### 3.1 Installation from PyPI

To install MindWare from PyPI, simply run the following command:

```bash
pip install mindware
```

### 3.2 Manual Installation from Source

To install MindWare using the source code, please run the following commands:

```bash
git clone https://github.com/thomas-young-2013/mindware.git && cd mindware
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

### 3.3 Test for Installation (TODO)

You can run the following code to test your installation:

```python
from mindware import run_test

if __name__ == '__main__':
    run_test()
```

If successful, you will receive the following message:

```
===== Congratulations! All trials succeeded. =====
```

If you encountered any problem during installation, please refer to the **Trouble Shooting** section.

## 4 Installation for Advanced Usage (Optional)

To use advanced features such as `pyrfr` (probabilistic random forest) surrogate in HPO, 
please refer to [Pyrfr Installation Guide](./install_pyrfr.md) to install `pyrfr`.

## 5 Trouble Shooting

If you encounter problems not listed below, please [File an issue](https://github.com/thomas-young-2013/mindware/issues) 
on GitHub or email us via *liyang.cs@pku.edu.cn*.

If you cannot install openbox correctly, please refer to [OpenBox Installation Guide](https://open-box.readthedocs.io/en/latest/installation/installation_guide.html).

### Windows

+ 'Error: \[WinError 5\] Access denied'. Please open the command prompt with administrative privileges or 
append `--user` to the command line.

+ 'ERROR: Failed building wheel for ConfigSpace'. Please refer to [tips](./install_configspace_on_win_fix_vc.md).

### macOS

+ For macOS users who have trouble installing pyrfr, please refer to [tips](./install-pyrfr-on-macos.md).

+ For macOS users who have trouble building scikit-learn, this [documentation](./openmp_macos.md) might help. 
