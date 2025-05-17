import json
import numpy as np

import torch

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def one_hot(value, num_classes):
    vec = np.zeros(num_classes)
    vec[value - 1] = 1
    return vec


def get_max_num_nodes(dataset_str):
    import datasets
    dataset = getattr(datasets, dataset_str)()

    max_num_nodes = -1
    for d in dataset.dataset:
        max_num_nodes = max(max_num_nodes, d.num_nodes)
    return max_num_nodes

def fill_nan_inf(a):
    a[np.isnan(a)] = 0.0
    a[np.isinf(a)] = 0.0
    return a
    
import torch
from torch_geometric.data import Batch


def is_nan_inf(x, name="x"):
    if torch.isnan(x).any():
        print(f'{name} is nan')
        return True
    if torch.isinf(x).any():
        print(f'{name} is inf')
        return True
    return False

def dense_to_edge_index(adj, is_sym=True, probability=None):
    if probability is not None:
        condition = lambda x: torch.nonzero(x > probability)
    else:
        condition = lambda x: torch.nonzero(x)

    if is_sym:
        tri_adj = torch.triu(adj, diagonal=1)
        edge_index = condition(tri_adj).T
    else:
        edge_index = condition(adj).T
    
    return edge_index


def edge_index_to_dense(edge_index, num_nodes=None, is_sym=True):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    if is_sym:
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
    else:
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
    return adj
    
        
def perturb_batch_edge_index(data, ratio, op):
    if op == 'rewire':
        return rewire_batch_edge_index(data, ratio)
    elif op == 'drop':
        return drop_batch_edge_index(data, ratio)
    else:
        raise NotImplementedError

def drop_batch_edge_index(data, ratio):
    # Number of graphs
    num_graphs = data.batch.max().item() + 1
    nodes_per_graph = torch.bincount(data.batch)
    ratio = max(min(ratio, 1), 0)
    
    new_edge_index = []
    
    for i in range(num_graphs):
        if nodes_per_graph[i] < 4:
            continue
        
        node_indices = torch.where(data.batch == i)[0]
        
        edge_indices = (data.edge_index[0] >= node_indices.min()) & (data.edge_index[0] <= node_indices.max())
        
        # Compute the number of edges to rewire
        total_edges = edge_indices.sum().item()
        num_edges_left = int(total_edges * (1-ratio))
        
        # Randomly select edges to rewire
        edges_left_indices = torch.where(edge_indices)[0][torch.randperm(total_edges)[:num_edges_left]]
        
        new_edge_inedx_left = data.edge_index[:, edges_left_indices]
        new_edge_index.append(new_edge_inedx_left)
     
    new_edge_index = torch.cat(new_edge_index, dim=1)
    
    return new_edge_index

def rewire_batch_edge_index(data, ratio):
   
    # Number of graphs
    num_graphs = data.batch.max().item() + 1
    nodes_per_graph = torch.bincount(data.batch)
    ratio = max(min(ratio, 1), 0)
    
    new_edge_index = data.edge_index.clone()
    for i in range(num_graphs):
        if nodes_per_graph[i] < 4:
            continue
        
        node_indices = torch.where(data.batch == i)[0]
        
        edge_indices = (data.edge_index[0] >= node_indices.min()) & (data.edge_index[0] <= node_indices.max())
        
        # Compute the number of edges to rewire
        total_edges = edge_indices.sum().item()
        num_edges_to_rewire = int(total_edges * ratio)
        
        # Randomly select edges to rewire
        edges_to_rewire = torch.where(edge_indices)[0][torch.randperm(total_edges)[:num_edges_to_rewire]]
        
        end_nodes = node_indices[torch.randint(node_indices.shape[0], (edges_to_rewire.shape[0], ))]
        new_edge_index[1, edges_to_rewire] = end_nodes.flatten()
        
     
    return new_edge_index

    
def adding_edge(node_num, edge_index, ratio=0.01):
    """
    Function to add edges to a given edge_index according to a provided ratio.

    Parameters:
    node_num (int): Number of nodes in the graph.
    edge_index (tensor): PyTorch tensor representing the original edge index.
    ratio (float): Ratio of edges to be added.

    Returns:
    edge_index (tensor): PyTorch tensor representing the added edge index.
    """
    # If no edge index is provided, return an empty tensor
    if edge_index is None or edge_index.nelement() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # Ensure the ratio is within a valid range
    ratio = max(min(ratio, 1), 0)

    # Compute the number of edges to add
    num_edges_to_add = int(edge_index.shape[1] * ratio)

    # Randomly select edges to add
    edges_to_add = torch.randint(node_num, (2, num_edges_to_add))

    # Create a copy of the edge_index to avoid modifying the original
    new_edge_index = edge_index.clone()

    # Add the selected edges
    new_edge_index = torch.cat([new_edge_index, edges_to_add], dim=1)
    
    # delete duplicated edges
    new_edge_index = torch.unique(new_edge_index, dim=1)

    # Return the added edge index
    return new_edge_index
    

def dropping_edge(edge_index, ratio=0.01):
   
    if edge_index is None or edge_index.nelement() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # Ensure the ratio is within a valid range
    ratio = max(min(ratio, 1), 0)

    # Compute the number of edges to drop
    num_edges_rest = int(edge_index.shape[1] * (1-ratio))
    # Randomly select edges to drop
    if num_edges_rest == edge_index.shape[1]:
        return edge_index
    else:
        print('dropped num: ', edge_index.shape[1] - num_edges_rest)
    
    edges_rest = torch.randperm(edge_index.shape[1])[:num_edges_rest]
    # Create a copy of the edge_index to avoid modifying the original
    new_edge_index = edge_index.clone()

    rest_edges = torch.stack([new_edge_index[0][edges_rest], new_edge_index[1][edges_rest]], dim=0)
    # Return the dropped edge index
    return rest_edges


def rewire_edge_index(N, edge_index, ratio=0.1):
    """
    Function to rewire a given edge_index according to a provided ratio.

    Parameters:
    N (int): Number of nodes in the graph.
    edge_index (tensor): PyTorch tensor representing the original edge index.
    ratio (float): Ratio of edges to be rewired.

    Returns:
    edge_index (tensor): PyTorch tensor representing the rewired edge index.
    """
    # If less than 2 nodes or no edge index is provided, return an empty tensor
    
    if N < 2 or edge_index is None or edge_index.nelement() == 0:
        print('Empty edge index!!!!!!!!!!!'*3)
        return torch.empty((2, 0), dtype=torch.long)

    # Ensure the ratio is within a valid range
    ratio = max(min(ratio, 1), 0)
    
    # Compute the number of edges to rewire
    num_edges_to_rewire = int(edge_index.shape[1] * ratio)
    
    # Randomly select edges to rewire
    edges_to_rewire = torch.randperm(edge_index.shape[1])[:num_edges_to_rewire]
    
    # Create a copy of the edge_index to avoid modifying the original
    new_edge_index = edge_index.clone()
    
    # Rewire the selected edges
    for edge in edges_to_rewire:
        # Avoid self-loops by generating new random end node different from the start node
        end_node = torch.randint(N, (1, ))
        while end_node == new_edge_index[0, edge]:
            end_node = torch.randint(N, (1, ))
        
        new_edge_index[1, edge] = end_node
    
    return new_edge_index


