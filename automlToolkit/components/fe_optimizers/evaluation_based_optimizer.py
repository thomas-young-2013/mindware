import time
import typing
from collections import namedtuple
from automlToolkit.components.feature_engineering.transformation_graph import *
from automlToolkit.components.fe_optimizers.base_optimizer import Optimizer
from automlToolkit.components.fe_optimizers.transformer_manager import TransformerManager
from automlToolkit.utils.decorators import time_limit, TimeoutException
from automlToolkit.components.evaluator import Evaluator
from automlToolkit.components.utils.constants import SUCCESS, ERROR, TIMEOUT

EvaluationResult = namedtuple('EvaluationResult', 'status duration score extra')


class EvaluationBasedOptimizer(Optimizer):
    def __init__(self, input_data: DataNode, evaluator: Evaluator,
                 model_id: str, time_limit_per_trans: int,
                 seed: int, shared_mode: bool = False,
                 batch_size: int = 2, beam_width: int = 3):
        super().__init__(str(__class__.__name__), input_data, seed)
        self.transformer_manager = TransformerManager()
        self.time_limit_per_trans = time_limit_per_trans
        self.evaluator = evaluator
        self.model_id = model_id
        self.incumbent_score = -1.
        self.baseline_score = -1.
        self.start_time = time.time()
        self.hp_config = None

        # Parameters in beam search.
        self.hpo_batch_size = batch_size
        self.beam_width = beam_width
        self.max_depth = 8
        self.trans_types = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
        # self.trans_types = [6, 7, 8, 9]

        self.iteration_id = 0
        self.evaluation_count = 0
        self.beam_set = list()
        self.is_ended = False
        self.evaluation_num_last_iteration = -1
        self.temporary_nodes = list()
        self.execution_history = dict()

        # Used to share new feature set.
        self.local_datanodes = list()
        self.global_datanodes = list()
        self.shared_mode = shared_mode

        # Avoid transformations, which would take too long
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
        execution_status = list()

        if self.iteration_id == 0:
            # Evaluate the original features.
            _start_time, status, extra = time.time(), SUCCESS, '%d,root_node' % _evaluation_cnt
            self.incumbent_score = self.evaluator(self.hp_config, data_node=self.root_node, name='fe')
            if self.incumbent_score is None:
                self.incumbent_score = 0.
                status = ERROR
            execution_status.append(EvaluationResult(status=status,
                                                     duration=time.time()-_start_time,
                                                     score=self.incumbent_score,
                                                     extra=extra))
            self.baseline_score = self.incumbent_score
            self.incumbent = self.root_node
            self.root_node.depth = 1
            _evaluation_cnt += 1
            self.beam_set.append(self.root_node)

        # Get one node in the beam set.
        node_ = self.beam_set[0]
        del self.beam_set[0]

        self.logger.debug('=' * 50)
        self.logger.info('Start %d-th FE iteration.' % self.iteration_id)

        # Limit the maximum depth in graph.
        # Avoid the too complex features.
        if node_.depth <= self.max_depth:
            # The polynomial and cross features are eliminated in the latter transformations.
            _trans_types = self.trans_types.copy()
            if node_.depth > 1 and 17 in _trans_types:
                _trans_types.remove(17)
            # Fetch available transformations for this node.
            trans_set = self.transformer_manager.get_transformations(
                node_, trans_types=_trans_types, batch_size=self.hpo_batch_size
            )
            self.logger.info('The number of transformations is: %d' % len(trans_set))
            for transformer in trans_set:
                self.logger.debug('[%s][%s]' % (self.model_id, transformer.name))

                if transformer.type != 0:
                    self.transformer_manager.add_execution_record(node_.node_id, transformer.type)

                _start_time, status, _score = time.time(), SUCCESS, -1
                extra = ['%d' % _evaluation_cnt, self.model_id, transformer.name]
                try:
                    # Limit the execution and evaluation time for each transformation.
                    with time_limit(self.time_limit_per_trans):
                        output_node = transformer.operate(node_)

                        # Evaluate this node.
                        if transformer.type != 0:
                            output_node.depth = node_.depth + 1
                            output_node.trans_hist.append(transformer.type)
                            _score = self.evaluator(self.hp_config, data_node=output_node, name='fe')
                            output_node.score = _score
                        else:
                            _score = output_node.score

                    if _score is None:
                        status = ERROR
                    else:
                        self.temporary_nodes.append(output_node)
                        self.graph.add_node(output_node)
                        # Avoid self-loop.
                        if transformer.type != 0 and node_.node_id != output_node.node_id:
                            self.graph.add_trans_in_graph(node_, output_node, transformer)
                        if _score > self.incumbent_score:
                            self.incumbent_score = _score
                            self.incumbent = output_node
                except Exception as e:
                    extra.append(str(e))
                    self.logger.error('%s: %s' % (transformer.name, str(e)))
                    status = ERROR
                    if isinstance(e, TimeoutException):
                        status = TIMEOUT

                execution_status.append(
                    EvaluationResult(status=status,
                                     duration=time.time() - _start_time,
                                     score=_score,
                                     extra=extra))
                _evaluation_cnt += 1

                self.evaluation_count += 1
                if (self.maximum_evaluation_num is not None
                    and self.evaluation_count > self.maximum_evaluation_num) or \
                        (self.time_budget is not None
                         and time.time() >= self.start_time + self.time_budget):
                    self.logger.debug('[Budget Runs Out]: %s, %s\n' % (self.maximum_evaluation_num, self.time_budget))
                    self.is_ended = True
                    break

        self.logger.info('\n [Current Inc]: %.4f, [Improvement]: %.5f' %
                         (self.incumbent_score, self.incumbent_score - self.baseline_score))

        self.evaluation_num_last_iteration = max(self.evaluation_num_last_iteration, _evaluation_cnt)

        # Update the beam set according to their performance.
        if len(self.beam_set) == 0:
            self.beam_set = list()
            self.local_datanodes = list()
            for node_ in TransformationGraph.sort_nodes_by_score(self.temporary_nodes)[:self.beam_width]:
                self.beam_set.append(node_)
                if self.shared_mode:
                    self.local_datanodes.append(node_)

            if self.shared_mode:
                self.logger.info('The number of local nodes: %d' % len(self.local_datanodes))
                self.logger.info('The local scores are: %s' % str([node.score for node in self.local_datanodes]))

            # Add the original dataset into the beam set.
            for _ in range(1 + self.beam_width - len(self.beam_set)):
                self.beam_set.append(self.root_node)
            self.temporary_nodes = list()
            self.logger.info('Finish one level in beam search: %d: %d' % (self.iteration_id, len(self.beam_set)))

        # Maintain the local incumbent data node.
        if self.shared_mode:
            if len(self.local_datanodes) == 0:
                self.local_datanodes.append(self.incumbent)
            if len(self.local_datanodes) > self.beam_width:
                self.local_datanodes = TransformationGraph.sort_nodes_by_score(self.local_datanodes)[:self.beam_width]

        self.iteration_id += 1
        self.execution_history[self.iteration_id] = execution_status
        iteration_cost = time.time() - _iter_start_time
        return self.incumbent.score, iteration_cost, self.incumbent

    def refresh_beam_set(self):
        if len(self.global_datanodes) > 0:
            self.logger.info('Sync the global nodes!')
            # Add local nodes.
            self.beam_set = self.local_datanodes[:self.beam_width-1]
            # Add Beam_size - 1 global nodes.
            for node in self.global_datanodes[:self.beam_width - 1]:
                self.beam_set.append(node)
