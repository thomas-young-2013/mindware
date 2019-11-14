import numpy as np
from .base_operator import UnaryOperator

unary_collection = ['log', 'sqrt', 'square', 'freq', 'round', 'tanh', 'sigmoid', 'zscore', 'norm']


class LogOperator(UnaryOperator):
    def __init__(self):
        self.operator_name = "LogOperator"

    def operate(self, col):
        return np.log(np.abs(col) + 1e-8)


class SqrtOperator(UnaryOperator):
    def __inif__(self):
        self.operator_name = "SqrtOperator"

    def operate(self, col):
        return np.sqrt(np.abs(col))


class SquareOperator(UnaryOperator):
    def __init__(self):
        self.operator_name = "SquareOperator"

    def operate(self, col):
        return np.square(col)


class FreqOperator(UnaryOperator):
    def __init__(self):
        self.operator_name = "FreqOperator"

    def operate(self, col):
        from collections import Counter
        counter = Counter(col)
        length = len(col)
        return np.array([counter[x] / length for x in col])


class RoundOperator(UnaryOperator):
    def __init__(self):
        self.operator_name = "RoundOperator"

    def operate(self, col):
        return np.around(col)


class TanhOperator(UnaryOperator):
    def __init__(self):
        self.operator_name = "TanhOperator"

    def operate(self, col):
        return np.tanh(col)


class SigmoidOperator(UnaryOperator):
    def __init__(self):
        self.operator_name = "SigmoidOperator"

    def operate(self, col):
        return self.sigmoid(col)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.array(x)))


class IsotonicOperator(UnaryOperator):
    def __init__(self):
        self.operator_name = "IsotonicOperator"

    def operate(self, col):
        from sklearn.isotonic import IsotonicRegression
        length = len(col)
        x = np.arange(length)
        ir = IsotonicRegression()
        return ir.fit_transform(x, col)


class ZscoreOperator(UnaryOperator):
    def __init__(self):
        self.operator_name = "ZscoreOperator"

    def operate(self, col):
        mean = np.mean(col)
        std = np.std(col)
        if std == 0:
            return [0] * len(col)
        else:
            return (col - mean) / std


class NormalizeOperator(UnaryOperator):
    def __init__(self):
        self.operator_name = "NormalizeOperator"

    def operate(self, col):
        x = np.array(col)
        max_val, min_val = max(col), min(col)
        if max_val == min_val:
            return [0.] * len(col)
        else:
            return (x - min_val) / (max_val - min_val)


op_dict = {'log': LogOperator(),
           'sqrt': SqrtOperator(),
           'square': SquareOperator(),
           'freq': FreqOperator(),
           'round': RoundOperator(),
           'tanh': TanhOperator(),
           'sigmoid': SigmoidOperator(),
           'isoreg': IsotonicOperator(),
           'zscore': ZscoreOperator(),
           'norm': NormalizeOperator()}
