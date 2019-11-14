import sklearn
import numpy as np
from .base_operator import BinaryOperator

binary_collection = ['sum', 'sub', 'multi', 'div']


class SumOperator(BinaryOperator):
    def __init__(self):
        self.operator_name = "SumOperator"

    def operate(self, col1, col2):
        return col1 + col2


class SubtractOperator(BinaryOperator):
    def __init__(self):
        self.operator_name = "SubtractOperator"

    def operate(self, col1, col2):
        return col1 - col2


class MultiplyOperator(BinaryOperator):
    def __init__(self):
        self.operator_name = "MultiplyOperator"

    def operate(self, col1, col2):
        return col1 * col2


class DivisionOperator(BinaryOperator):
    def __init__(self):
        self.operator_name = "DivisionOperator"

    def operate(self, col1, col2):
        return col1 / col2
