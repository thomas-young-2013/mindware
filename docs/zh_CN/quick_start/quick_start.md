# 快速入门

本教程将指导您运行第一个 **MindWare** 程序。

## 数据准备

首先，为端到端自动化机器学习系统 **准备数据**。
这里我们用sklearn中的iris数据集。

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
```

在导入并切分数据集后，使用 **DataManager** 包装数据。

```python
from mindware.utils.data_manager import DataManager

dm = DataManager(X_train, y_train)
train_data = dm.get_data_node(X_train, y_train)
test_data = dm.get_data_node(X_test, y_test)
```

## 优化

**MindWare** 提供了一种便捷的、端到端的方式，来完成机器学习任务。
在这个例子中，我们使用 <font color=#FF0000>**Classifier**</font> 来解决分类任务。
请指定优化的时间约束 **time_limit**，然后只需调用<font color=#FF0000>**Classifier.fit**</font>，
系统就会自动执行特征工程、模型选择、超参数优化和模型集成过程。
对于大搜索空间，系统还会通过搜索空间分解来加速优化过程。

```python
from mindware.estimators import Classifier

clf = Classifier(time_limit=3600)
clf.fit(train_data)
```

优化结束后，调用 <font color=#FF0000>**Classifier.predict**</font> 来获取自动化机器学习系统对于测试数据集的预测结果。

```python
pred = clf.predict(test_data)
```

(todo: result and visualization)
