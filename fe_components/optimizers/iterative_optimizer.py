from fe_components.transformation_graph import *
from fe_components.transformers.merger import Merger
from fe_components.optimizers.base_optimizer import Optimizer
from fe_components.transformers import _transformers, _type_infos, _params_infos
from fe_components.transformers.model_based_selector import ModelBasedSelector
from fe_components.transformers.polynomial_generator import PolynomialTransformation


class IterativeOptimizer(Optimizer):
    def __init__(self, input_data: DataNode):
        super().__init__(str(__class__), input_data)

    def optimize(self):
        self.iterate()
        return self.incumbent

    def generate(self):
        feature_types = self.root_node.feature_types
        input_node = self.root_node
        self.graph.add_node(input_node)
        generated_nodes = list()

        # Apply possible transformations to the raw dataset.
        for feature_type in set(feature_types):
            transformer_ids = _type_infos[feature_type]
            transformers = list()

            for id in transformer_ids:
                if _transformers[id]().type not in [1, 4, 7]:
                    continue

                params = _params_infos[id]
                if len(params) == 0:
                    transformers.append(_transformers[id]())
                else:
                    for param in params:
                        transformer = _transformers[id](param=param)
                        transformers.append(transformer)

            print('#transformations', len(transformers))
            for transformer in transformers:
                output_node = transformer.operate(input_node)
                generated_nodes.append(output_node)
                self.graph.add_node(output_node)
                self.graph.add_trans_in_graph(input_node, output_node, transformer)

        # Apply cross transformations on the categorical features.
        if input_node.cat_num > 0:
            transformer = PolynomialTransformation()
            transformer.input_type = CATEGORICAL
            transformer.output_type = CATEGORICAL
            output_node = transformer.operate(input_node)
            generated_nodes.append(output_node)
            self.graph.add_node(output_node)
            self.graph.add_trans_in_graph(input_node, output_node, transformer)

        transformer = Merger()
        output_node = transformer.operate(generated_nodes)
        self.graph.add_node(output_node)
        self.graph.add_trans_in_graph(generated_nodes, output_node, transformer)

    def rank(self):
        input_node = self.graph.get_node(self.graph.node_size - 1)

        transformer = Merger()
        input_nodes = [input_node, self.root_node]
        output_node = transformer.operate(input_nodes)
        self.graph.add_node(output_node)
        self.graph.add_trans_in_graph(input_nodes, output_node, transformer)

        input_node = output_node
        transformer = ModelBasedSelector(max_features=input_node.data[0].shape[1])
        output_node = transformer.operate(input_node)
        self.graph.add_node(output_node)
        self.graph.add_trans_in_graph(input_node, output_node, transformer)
        self.incumbent = output_node

    def iterate(self):
        self.generate()
        self.rank()
