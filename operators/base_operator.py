class Operator(object):
    def __init__(self):
        self.operator_name = None


class UnaryOperator(Operator):
    def operate(self, col):
        raise NotImplementedError()


class BinaryOperator(Operator):
    def operate(self, col1, col2):
        raise NotImplementedError()
