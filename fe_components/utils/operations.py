import numpy as np

class Arithmetic:
    def fit(self, array):
        return

    def transform(self, array):
        raise NotImplementedError()


# Abstract log
class Log(Arithmetic):
    def transform(self, array):
        return np.log(np.abs(array) + 1e-8)


# Abstract sqrt
class Sqrt(Arithmetic):
    def transform(self, array):
        return np.sqrt(np.abs(array))


class Square(Arithmetic):
    def transform(self, array):
        return np.square(array)


class Freq(Arithmetic):
    def __init__(self):
        self.hashmap = []
        self.length = None
        self.col_num = None

    def fit(self, array):
        from collections import Counter
        self.col_num = array.shape[1]
        self.length = array.shape[0]
        counters = []
        for i in range(self.col_num):
            counters.append(Counter(array[:, i]))
        for ct in counters:
            self.hashmap.append({x: ct[x] / self.length for x in ct})

    def transform(self, array):
        result = []
        col_num = array.shape[1]
        assert (self.col_num == col_num)
        for i in range(col_num):
            col_result = []
            for x in array[:, i]:
                if x in self.hashmap[i]:
                    col_result.append(self.hashmap[i][x])
                else:
                    col_result.append(1 / self.length)
            result.append(col_result)
        return np.array(result).transpose()


class Round(Arithmetic):
    def transform(self, array):
        return np.around(array)


class Tanh(Arithmetic):
    def transform(self, array):
        return np.tanh(array)


class Sigmoid(Arithmetic):
    def transform(self, array):
        return 1 / (1 + np.exp(-np.array(array)))

class BinaryArithmetic:
    def fit(self, array1, array2):
        return

    def transform(self, array1, array2):
        raise NotImplementedError()


class Addition(BinaryArithmetic):
    def transform(self, array1, array2):
        return array1 + array2


class Subtract(BinaryArithmetic):
    def transform(self, array1, array2):
        return array1 - array2


class Multiply(BinaryArithmetic):
    def transform(self, array1, array2):
        return array1 * array2


class Division(BinaryArithmetic):
    def transform(self, array1, array2):
        return array1 / array2

