import time
from fe_components.transformation_graph import *
from fe_components.optimizers.base_optimizer import Optimizer
from fe_components.transformers.polynomial_generator import PolynomialTransformation
from fe_components.transformers.variance_selector import VarianceSelector
from fe_components.transformers.model_based_selector import ModelBasedSelector
from fe_components.transformers.pca_decomposer import PcaDecomposer


class EvaluationBasedOptimizer(Optimizer):
    def __init__(self, input_data: DataNode, evaluator):
        super().__init__(str(__class__), input_data)
        self.evaluator = evaluator
        self.incumbent_score = -1.
        self.start_time = time.time()

    def optimize(self):
        # Evaluate the original features.
        root_score = self.evaluator(self.root_node)
        self.incumbent_score = root_score
        self.incumbent = self.root_node

        num_limit = self.maximum_evaluation_num if self.maximum_evaluation_num is not None else 100000
        budget_limit = self.time_budget
        max_depth = 5
        beam_width = 2

        from queue import Queue
        queue = Queue()

        cnt = 0
        self.root_node.depth = 1

        queue.put(self.root_node)

        # The implementation of Beam Search (https://en.wikipedia.org/wiki/Beam_search).
        is_ended = False
        while not queue.empty() and not is_ended:
            node_ = queue.get()

            # Limit the maximum depth in graph.
            if node_.depth > max_depth:
                break

            # Fetch available transformations for this node.
            trans_types = [1, 2, 3, 4, 8, 9]
            # trans_types = [9]
            trans_set = self.get_available_transformations(node_, trans_types=trans_types)
            nodes = list()

            for transformer in trans_set:
                if transformer.type not in [9]:
                    transformer.compound_mode = 'in_place'
                output_node = transformer.operate(node_)
                output_node.depth = node_.depth + 1
                nodes.append(output_node)
                # Evaluate this node.
                _score = self.evaluator(output_node)
                output_node.score = _score
                if _score > self.incumbent_score:
                    self.incumbent_score = _score
                    self.incumbent = output_node

                self.graph.add_node(output_node)
                self.graph.add_trans_in_graph(node_, output_node, transformer)
                cnt += 1
                if cnt > num_limit or (budget_limit is not None and time.time() >= self.start_time + budget_limit):
                    print('==> Budget runs out!', num_limit, budget_limit)
                    is_ended = True
                    break

            for node_ in TransformationGraph.sort_nodes_by_score(nodes)[:beam_width]:
                queue.put(node_)

            print('==> Current incumbent', self.incumbent_score, 'Improvement: ', self.incumbent_score - root_score)

        try:
            input_node = self.incumbent
            transformer = VarianceSelector()
            output_node = transformer.operate(input_node)
            self.graph.add_node(output_node)
            self.graph.add_trans_in_graph(input_node, output_node, transformer)
            _score = self.evaluator(output_node)
            if _score >= self.incumbent_score:
                self.incumbent_score = _score
                self.incumbent = output_node
        except ValueError as e:
            print(e)
            return self.incumbent

        # 1. Apply polynomial transformation on the non-categorical features.
        # 2. Conduct feature selection.
        if input_node.shape[1] - input_node.cat_num > 1:
            input_node = self.incumbent
            transformer = PolynomialTransformation()
            transformer.compound_mode = 'concatenate'
            try:
                output_node = transformer.operate(input_node)
                print('Shape ==>', input_node.shape, output_node.shape)
                self.graph.add_node(output_node)
                self.graph.add_trans_in_graph(input_node, output_node, transformer)

                input_node = output_node
                transformer = ModelBasedSelector(param='et')
                output_node = transformer.operate(input_node)
                print('Shape ==>', input_node.shape, output_node.shape)
                self.graph.add_node(output_node)
                self.graph.add_trans_in_graph(input_node, output_node, transformer)

                _score = self.evaluator(output_node)

                if _score > self.incumbent_score:
                    self.incumbent_score = _score
                    self.incumbent = output_node
            except MemoryError as e:
                print(e)

        # 1. Apply cross transformations on the categorical features.
        # 2. Conduct feature selection.
        if input_node.cat_num > 1:
            input_node = self.incumbent
            transformer = PolynomialTransformation()
            transformer.compound_mode = 'concatenate'
            transformer.input_type = CATEGORICAL
            transformer.output_type = CATEGORICAL
            try:
                output_node = transformer.operate(input_node)
                print('Shape ==>', input_node.shape, output_node.shape)
                self.graph.add_node(output_node)
                self.graph.add_trans_in_graph(input_node, output_node, transformer)

                input_node = output_node
                transformer = ModelBasedSelector(param='et')
                output_node = transformer.operate(input_node)
                print('Shape ==>', input_node.shape, output_node.shape)
                self.graph.add_node(output_node)
                self.graph.add_trans_in_graph(input_node, output_node, transformer)

                _score = self.evaluator(output_node)

                if _score > self.incumbent_score:
                    self.incumbent_score = _score
                    self.incumbent = output_node
            except MemoryError as e:
                print(e)

        # PCA Dimension reduction.
        if True:
            input_node = self.incumbent
            transformer = PcaDecomposer()
            output_node = transformer.operate(input_node)
            print('Shape ==>', input_node.shape, output_node.shape)
            self.graph.add_node(output_node)
            self.graph.add_trans_in_graph(input_node, output_node, transformer)

            _score = self.evaluator(output_node)

            if _score > self.incumbent_score:
                self.incumbent_score = _score
                self.incumbent = output_node
        return self.incumbent
