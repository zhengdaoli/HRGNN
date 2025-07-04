from collections import defaultdict
import numpy as np
import networkx as nx

from .graph import Graph
from utils.encode_utils import one_hot


def parse_tu_data(name, raw_dir):
    # setup paths
    indicator_path = raw_dir / name / f'{name}_graph_indicator.txt'
    edges_path = raw_dir / name / f'{name}_A.txt'
    graph_labels_path = raw_dir / name / f'{name}_graph_labels.txt'
    node_labels_path = raw_dir / name / f'{name}_node_labels.txt'
    edge_labels_path = raw_dir / name / f'{name}_edge_labels.txt'
    node_attrs_path = raw_dir / name / f'{name}_node_attributes.txt'
    edge_attrs_path = raw_dir / name / f'{name}_edge_attributes.txt'

    unique_node_labels = set()
    unique_edge_labels = set()

    indicator, edge_indicator = [-1], [(-1,-1)]
    graph_nodes = defaultdict(list)
    graph_edges = defaultdict(list)
    node_labels = defaultdict(list)
    edge_labels = defaultdict(list)
    node_attrs = defaultdict(list)
    edge_attrs = defaultdict(list)
    
    total_nodes_num = 0
    with open(indicator_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            if len(line.strip()) < 1:
                continue
            line = line.rstrip("\n")
            graph_id = int(line)
            indicator.append(graph_id)
            graph_nodes[graph_id].append(i)
            total_nodes_num += 1
    print('total node num:', total_nodes_num)
    
    
    
    # num_nodes = num_nodes_map[name] 
    num_nodes = total_nodes_num
    nodes = np.arange(1, num_nodes+1)
    
    G = nx.Graph() # NOTE: holistic graph combined all graphs.
    G.add_nodes_from(nodes)
    
    with open(edges_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            edge = [int(e) for e in line.split(',')]
            edge_indicator.append(edge)

            # edge[0] is a node id, and it is used to retrieve
            # the corresponding graph id to which it belongs to
            # (see README.txt)
            graph_id = indicator[edge[0]]

            graph_edges[graph_id].append(edge)
            
            G.add_edge(edge[0], edge[1])

    if node_labels_path.exists():
        with open(node_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                node_label = int(line)
                unique_node_labels.add(node_label)
                graph_id = indicator[i]
                node_labels[graph_id].append(node_label)

    if edge_labels_path.exists():
        with open(edge_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                edge_label = int(line)
                unique_edge_labels.add(edge_label)
                graph_id = indicator[edge_indicator[i][0]]
                edge_labels[graph_id].append(edge_label)

    if node_attrs_path.exists():
        with open(node_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split(",")
                node_attr = np.array([float(n) for n in nums])
                graph_id = indicator[i]
                node_attrs[graph_id].append(node_attr)

    if edge_attrs_path.exists():
        with open(edge_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split(",")
                edge_attr = np.array([float(n) for n in nums])
                graph_id = indicator[edge_indicator[i][0]]
                edge_attrs[graph_id].append(edge_attr)

    # get graph labels
    graph_labels = []
    with open(graph_labels_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            target = int(line)
            if target == -1:
                graph_labels.append(0)
            else:
                graph_labels.append(target)

        if min(graph_labels) == 1:  # Shift by one to the left. Apparently this is necessary for multiclass tasks.
            graph_labels = [l - 1 for l in graph_labels]

    num_node_labels = max(unique_node_labels) if unique_node_labels != set() else 0
    if num_node_labels != 0 and min(unique_node_labels) == 0:  # some datasets e.g. PROTEINS have labels with value 0
        num_node_labels += 1

    num_edge_labels = max(unique_edge_labels) if unique_edge_labels != set() else 0
    if num_edge_labels != 0 and min(unique_edge_labels) == 0:
        num_edge_labels += 1

    return {
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "graph_labels": graph_labels,
        "node_labels": node_labels,
        "node_attrs": node_attrs,
        "edge_labels": edge_labels,
        "edge_attrs": edge_attrs
    }, num_node_labels, num_edge_labels, G


def create_graph_from_nx(nx_graph:nx.Graph, label) -> Graph:
    G = Graph(target=label)
    
    for n in nx_graph.nodes:
        G.add_node(n, label=None, attrs=None)
    
    for (n1, n2) in nx_graph.edges:
        G.add_edge(n1, n2, label=None, attrs=None)
    
    #TODO: if has node attr or edge attr:
    
    return G
    
    
def create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels, Graph_whole):
    # Graph is the networks graph containing all nodes and edges in the dataset
    nodes = graph_data["graph_nodes"]
    edges = graph_data["graph_edges"]

    G = Graph(target=target)

    for i, node in enumerate(nodes):
        label, attrs = None, None

        if graph_data["node_labels"] != []:
            label = one_hot(graph_data["node_labels"][i], num_node_labels)

        if graph_data["node_attrs"] != []:
            attrs = graph_data["node_attrs"][i]

        G.add_node(node, label=label, attrs=attrs)
    
    for i, edge in enumerate(edges):
        n1, n2 = edge
        label, attrs = None, None

        if graph_data["edge_labels"] != []:
            label = one_hot(graph_data["edge_labels"][i], num_edge_labels)
        if graph_data["edge_attrs"] != []:
            attrs = graph_data["edge_attrs"][i]

        G.add_edge(n1, n2, label=label, attrs=attrs)

    return G

def get_dataset_node_num(dataset_name):
    num_nodes_map = {
        "MUTAG": 3371,
        "ENZYMES": 19580, 
        "IMDB-BINARY": 19773, 
        "IMDB-MULTI": 19502, 
        "DD": 334925,
        "PROTEINS_full": 43471,
        "COLLAB": 372474,
    }

    return num_nodes_map[dataset_name]