import numpy as np
from automlToolkit.components.utils.constants import CATEGORICAL


class DataNode(object):
    def __init__(self, data=None, feature_type=None, task_type=None):
        self.task_type = task_type
        self.data = data
        self.feature_types = feature_type
        self._node_id = -1
        self.depth = None
        self.score = None
        self.trans_hist = list()
        self.enable_balance = False

    def __eq__(self, node):
        """Overrides the default implementation"""
        if isinstance(node, DataNode):
            if self.shape != node.shape:
                return False
            X1 = self.data[0].astype("float64")
            X2 = node.data[0].astype("float64")
            X_flag = np.isclose(X1, X2).all()
            y_flag = np.isclose(self.data[1], node.data[1]).all()
            if X_flag and y_flag:
                return True
        return False

    def __add__(self, other):
        X1, y1 = self.copy_().data
        X2, y2 = other.copy_().data
        feat_types = self.feature_types.copy()
        X = np.vstack((X1, X2))
        y = np.vstack((y1, y2))
        return DataNode(data=[X, y], feature_type=feat_types)

    def copy_(self):
        new_data = list([self.data[0].copy()])
        new_data.append(None if self.data[1] is None else self.data[1].copy())
        new_node = DataNode(new_data, self.feature_types.copy(), self.task_type)
        new_node.trans_hist = self.trans_hist.copy()
        new_node.depth = self.depth
        return new_node

    def set_values(self, node):
        """ Assign node's content to current node.

        Assign the variables "data, feature_types, and task_type" of node to the current node.
        This function does NOT assign the node id.

        :param node: the data node is copied.
        :return: None.
        """
        self.data = []
        for val in node.data[:2]:
            self.data.append(val.copy() if val is not None else None)
        self.feature_types = node.feature_types.copy()
        self.task_type = node.task_type

    @property
    def node_id(self):
        return self._node_id

    @property
    def cat_num(self):
        cnt = 0
        for feature_type in self.feature_types:
            cnt += 1 if feature_type == CATEGORICAL else 0
        return cnt

    @property
    def shape(self):
        assert self.data[0].shape[1] == len(self.feature_types)
        return self.data[0].shape

    def __str__(self):
        from tabulate import tabulate
        tabular_data = list()
        if len(self.feature_types) > 8:
            types_summary = ','.join(self.feature_types[:4])
            types_summary += ',...,' + ','.join(self.feature_types[-4:])
        else:
            types_summary = ','.join(self.feature_types)
        tabular_data.append(['feature types', types_summary])
        tabular_data.append(['data shape', '%d, %d' % self.shape])
        tabular_data.append(['#Cat-feature', self.cat_num])
        tabular_data.append(['#NonCat-feature', self.shape[1] - self.cat_num])
        return tabulate(tabular_data, tablefmt="github")


class TransformationEdge(object):
    def __init__(self, input, output, transformer, fields):
        self.id = -1
        self.input_id = input
        self.output_id = output
        self.target_fields = fields
        self.transformer = transformer


class TransformationGraph(object):
    def __init__(self):
        # Store the data nodes.
        self.nodes = list()
        # Store the edge information.
        self.edges = list()
        self.node_size = 0
        self.edge_size = 0
        self.input_data_dict = dict()
        self.input_edge_dict = dict()
        self.adjacent_list = dict()

    def add_edge(self, input, output, transformer):
        fields = transformer.target_fields
        edge = TransformationEdge(input, output, transformer, fields)
        edge.id = self.edge_size
        self.edges.append(edge)
        self.edge_size += 1
        if output not in self.input_data_dict:
            self.input_data_dict[output] = list()
        self.input_data_dict[output].append(input)

        if input not in self.adjacent_list:
            self.adjacent_list[input] = list()
        self.adjacent_list[input].append(output)

        if output not in self.input_edge_dict:
            self.input_edge_dict[output] = edge.id

    def add_trans_in_graph(self, input_datanode, output_datanode, transformer):
        if type(input_datanode) is not list:
            input_ids = [input_datanode.node_id]
        else:
            input_ids = [node.node_id for node in input_datanode]

        for input_id in input_ids:
            self.add_edge(input_id, output_datanode.node_id, transformer)

    def add_node(self, data_node: DataNode):
        # Avoid adding the same node into the graph multiple times.
        if data_node.node_id != -1:
            return data_node.node_id

        node_id = self.node_size
        data_node._node_id = node_id
        image_node = data_node.copy_()
        # Image node does not store the data in the graph.
        image_node.data = None
        image_node._node_id = node_id
        self.nodes.append(image_node)
        self.node_size += 1
        return node_id

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_edge(self, edge_id):
        return self.edges[edge_id]

    def topological_sort(self):
        for i in range(self.node_size):
            if i not in self.adjacent_list:
                self.adjacent_list[i] = list()

        is_visited = [False] * self.node_size
        result = []

        def dfs(v, visited, stack):
            visited[v] = True
            for i in self.adjacent_list[v]:
                if not visited[i]:
                    dfs(i, visited, stack)
            stack.insert(0, v)

        for i in range(self.node_size):
            if not is_visited[i]:
                dfs(i, is_visited, result)

        return result

    def get_path_nodes(self, node: DataNode):
        result = set()

        def traverse(node_id):
            result.add(node_id)
            if node_id in self.input_data_dict:
                for parent_id in self.input_data_dict[node_id]:
                    traverse(parent_id)

        traverse(node.node_id)
        orders = self.topological_sort()

        path_ids = list()
        for node_id in orders:
            if node_id in result:
                path_ids.append(node_id)
        return path_ids

    @staticmethod
    def sort_nodes_by_score(nodes: DataNode):
        return sorted(nodes, key=lambda node: -node.score)
