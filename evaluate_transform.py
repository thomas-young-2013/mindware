from components.transformers.preprocessor.imputer import ImputationTransformation
from components.feature_engineering.transformations.continous_discretizer import *
from components.feature_engineering.transformations.rescaler.scaler import ScaleTransformation
from components.transformers.preprocessor.onehot_encoder import OneHotTransformation
from components.feature_engineering.transformations.merger import Merger
from components.feature_engineering.transformation_graph import DataNode
from components.feature_engineering.fe_pipeline import FEPipeline
from components.utils.constants import *
from components.feature_engineering.transformation_graph import TransformationGraph


def evaluate_transformation_graph():
    data = (np.array([
        [np.nan, 2, 1],
        [1, 2, 2],
        [3, 4, 2],
        [5, np.nan, 1]
    ]), np.array([1, 2, 3, 4]))
    feature_type = [NUMERICAL, NUMERICAL, CATEGORICAL]
    datanode = DataNode(data, feature_type)

    graph = TransformationGraph()
    graph.add_node(datanode)

    transformer = ImputationTransformation('most_frequent')
    output_datanode1 = transformer.operate(datanode, target_fields=[0, 1])
    graph.add_node(output_datanode1)
    graph.add_edge(datanode.get_node_id(), output_datanode1.get_node_id(), transformer)

    transformer = OneHotTransformation()
    output_datanode2 = transformer.operate(output_datanode1)
    graph.add_node(output_datanode2)
    graph.add_edge(output_datanode1.get_node_id(), output_datanode2.get_node_id(), transformer)

    transformer = ScaleTransformation(param='standard')
    transformer.concatenate = True
    output_datanode3 = transformer.operate(output_datanode2)
    graph.add_node(output_datanode3)
    graph.add_edge(output_datanode2.get_node_id(), output_datanode3.get_node_id(), transformer)

    print(output_datanode3)
    print(output_datanode3.data)

    transformer = ScaleTransformation(param='min_max')
    transformer.concatenate = False
    output_datanode4 = transformer.operate(output_datanode2)
    graph.add_node(output_datanode4)
    graph.add_edge(output_datanode2.get_node_id(), output_datanode4.get_node_id(), transformer)

    transformer = Merger()
    output_datanode5 = transformer.operate([output_datanode3, output_datanode4])
    graph.add_node(output_datanode5)
    graph.add_transformation([output_datanode3.get_node_id(), output_datanode4.get_node_id()], output_datanode5.get_node_id(), transformer)

    print(output_datanode5)
    print(output_datanode5.data)

    order_ids = graph.topological_sort()
    print(order_ids)
    test_data = (np.array([
        [np.nan, 2, 1],
        [1, 2, 1],
        [3, 2, 1],
        [3, np.nan, 1]
    ]), None)
    test_node = DataNode(test_data, feature_types)

    inputnode = graph.get_node(order_ids[0])
    inputnode.set_values(test_node)

    for idx in range(1, len(order_ids)):
        node_id = order_ids[idx]

        input_node_list = list()
        for input_id in graph.input_data_dict[node_id]:
            inputnode = graph.get_node(input_id)
            input_node_list.append(inputnode)
        inputnode = input_node_list[0] if len(input_node_list) == 1 else input_node_list

        edge = graph.get_edge(graph.input_edge_dict[node_id])
        outputnode = edge.transformer.operate(inputnode, edge.target_fields)
        graph.get_node(node_id).set_values(outputnode)
    output_node = graph.get_node(order_ids[-1])
    print(output_node)
    print(output_node.data)


def evaluate_fe_pipeline():
    from utils.data_manager import DataManager
    dm = DataManager()
    # file_path = "data/proprocess_data.csv"
    file_path = 'data/a9a/dataset_183_adult.csv'
    dm.load_train_csv(file_path)

    pipeline = FEPipeline(fe_enabled=True).fit(dm)
    train_data = pipeline.transform(dm)
    print(train_data)
    print(train_data.data)

    # test_dm = DataManager()
    # test_dm.load_train_csv(file_path)
    # test_dm.train_X = dm.train_X
    # test_dm.train_y = dm.train_y
    #
    # test_data = pipeline.transform(test_dm)
    # assert (train_data.data[0] == test_data.data[0]).all()


def evaluate_data_manager():
    # from data_manager import DataManager
    # dm = DataManager()
    # train_df = dm.load_train_csv("data/proprocess_data.csv")
    # print(train_df)
    # print(dm.feature_types)
    # print(dm.missing_flags)

    from utils.data_manager import DataManager
    import numpy as np
    X = np.array([[1, 2, 3, 4], [1, 'asfd', 2, 1.4]])
    y = [1, 2]
    dm = DataManager(X, y)
    print(dm.feature_types)
    print(dm.missing_flags)


if __name__ == "__main__":
    evaluate_fe_pipeline()
