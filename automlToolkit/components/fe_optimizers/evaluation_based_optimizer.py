import time
from automlToolkit.components.feature_engineering.transformation_graph import *
from automlToolkit.components.fe_optimizers.base_optimizer import Optimizer
from automlToolkit.components.fe_optimizers.transformer_manager import TransformerManager


class EvaluationBasedOptimizer(Optimizer):
    def __init__(self, input_data: DataNode, evaluator, model_id, seed):
        super().__init__(str(__class__.__name__), input_data, seed)
        self.transformer_manager = TransformerManager()
        self.evaluator = evaluator
        self.incumbent_score = -1.
        self.baseline_score = -1.
        self.start_time = time.time()
        self.hp_config = None

        self.max_depth = 8
        self.beam_width = 3
        self.iteration_id = 0
        self.evaluation_count = 0
        self.beam_set = list()
        self.is_ended = False
        self.evaluation_num_last_iteration = -1
        self.temporary_nodes = list()

        self.trans_types = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
        # self.trans_types = [0, 3, 4, 5, 6, 7, 8, 9]

        # which would take too long
        # Combinations of non-linear models with feature learning.
        # feature_learning = ["kitchen_sinks", "kernel_pca", "nystroem_sampler"]
        classifier_set = ["adaboost", "decision_tree", "extra_trees",
                          "gradient_boosting", "k_nearest_neighbors",
                          "libsvm_svc", "random_forest", "gaussian_nb", "decision_tree"]

        if model_id in classifier_set:
            for tran_id in [12, 13, 15]:
                if tran_id in self.trans_types:
                    self.trans_types.remove(tran_id)

    def optimize(self):
        while not self.is_ended:
            self.logger.debug('='*50)
            self.logger.debug('Start the ITERATION: %d' % self.iteration_id)
            self.logger.debug('='*50)
            self.iterate()
        return self.incumbent

    def iterate(self):
        _iter_start_time = time.time()
        _evaluation_cnt = 0
        if self.iteration_id == 0:
            # Evaluate the original features.
            self.incumbent_score = self.evaluator(self.hp_config, data_node=self.root_node, name='fe')
            if self.incumbent_score is None:
                self.incumbent_score = 0.
            self.baseline_score = self.incumbent_score
            self.incumbent = self.root_node
            self.root_node.depth = 1
            _evaluation_cnt += 1
            self.beam_set.extend([self.root_node] * (self.beam_width + 1))

        # Get one node in the beam set.
        node_ = self.beam_set[0]
        del self.beam_set[0]

        self.logger.debug('=' * 50)
        self.logger.info('Start %d-th FE iteration.' % self.iteration_id)

        # Limit the maximum depth in graph.
        # Avoid the too complex features.
        if node_.depth > self.max_depth:
            self.iteration_id += 1
            iteration_cost = time.time() - _iter_start_time
            return self.incumbent.score, iteration_cost, self.incumbent

        # The polynomial and cross features are eliminated in the latter transformations.
        _trans_types = self.trans_types.copy()
        if node_.depth > 1 and 17 in _trans_types:
            _trans_types.remove(17)
        # Fetch available transformations for this node.
        trans_set = self.transformer_manager.get_transformations(node_, trans_types=_trans_types)

        for transformer in trans_set:
            # Avoid repeating the same transformation multiple times.
            if transformer.type in node_.trans_hist:
                continue

            error_msg = None
            try:
                self.logger.debug('[%s]' % transformer.name)
                output_node = transformer.operate(node_)
                output_node.depth = node_.depth + 1
                output_node.trans_hist.append(transformer.type)

                # Evaluate this node.
                _score = self.evaluator(self.hp_config, data_node=output_node, name='fe')
                output_node.score = _score
                if _score is not None and _score > self.incumbent_score:
                    self.incumbent_score = _score
                    self.incumbent = output_node

                if _score is not None:
                    self.temporary_nodes.append(output_node)
                    self.graph.add_node(output_node)
                    self.graph.add_trans_in_graph(node_, output_node, transformer)
            except ValueError as e:
                error_msg = '%s: %s' % (transformer.name, str(e))
            except MemoryError as e:
                error_msg = '%s: %s' % (transformer.name, str(e))
            except RuntimeError as e:
                error_msg = '%s: %s' % (transformer.name, str(e))
            except IndexError as e:
                error_msg = '%s: %s' % (transformer.name, str(e))
            finally:
                if error_msg is not None:
                    self.logger.error(error_msg)

            _evaluation_cnt += 1

            self.evaluation_count += 1
            if (self.maximum_evaluation_num is not None
                and self.evaluation_count > self.maximum_evaluation_num) or \
                    (self.time_budget is not None
                     and time.time() >= self.start_time + self.time_budget):
                self.logger.debug('[Budget Runs Out]: %s, %s\n' % (self.maximum_evaluation_num, self.time_budget))
                self.is_ended = True
                break

        self.logger.info('\n [Current Inc]: %.4f, [Improvement]: %.5f'
                          % (self.incumbent_score, self.incumbent_score - self.baseline_score))

        self.evaluation_num_last_iteration = max(self.evaluation_num_last_iteration, _evaluation_cnt)

        # Update the beam set according to their performance.
        if len(self.beam_set) == 0:
            self.beam_set = list()
            for node_ in TransformationGraph.sort_nodes_by_score(self.temporary_nodes)[:self.beam_width]:
                self.beam_set.append(node_)
            # Add the original dataset into the beam set.
            for _ in range(1 + self.beam_width - len(self.beam_set)):
                self.beam_set.append(self.root_node)
            self.temporary_nodes = list()
            self.logger.info('Finish one level in beam search: %d' % self.iteration_id)

        self.iteration_id += 1
        iteration_cost = time.time() - _iter_start_time
        return self.incumbent.score, iteration_cost, self.incumbent
