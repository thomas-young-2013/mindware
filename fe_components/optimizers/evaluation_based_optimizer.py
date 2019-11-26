import time
from fe_components.transformation_graph import *
from fe_components.optimizers.base_optimizer import Optimizer
from fe_components.transformers.generator.polynomial_generator import PolynomialTransformation
from fe_components.transformers.selector.variance_selector import VarianceSelector
from fe_components.transformers.selector.model_based_selector import ModelBasedSelector
from fe_components.transformers.generator.pca_decomposer import PcaDecomposer


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

        num_limit = self.maximum_evaluation_num if self.maximum_evaluation_num is not None else 10000000
        budget_limit = self.time_budget
        max_depth = 100000
        beam_width = 3

        cnt = 0
        self.root_node.depth = 1

        # The implementation of Beam Search (https://en.wikipedia.org/wiki/Beam_search).
        is_ended = False
        beam_set = [self.root_node]
        while len(beam_set) > 0 and not is_ended:
            nodes = list()
            for node_ in beam_set:
                # Limit the maximum depth in graph.
                if node_.depth > max_depth:
                    break

                # Fetch available transformations for this node.
                # trans_types = [1, 2, 3, 4, 5, 8, 9]
                trans_types = list(range(20))
                trans_set = self.get_available_transformations(node_, trans_types=trans_types)

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

            beam_set = list()
            for node_ in TransformationGraph.sort_nodes_by_score(nodes)[:beam_width]:
                beam_set.append(node_)

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
