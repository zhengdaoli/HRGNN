import networkx as nx

import torch
from torch_geometric.utils import dense_to_sparse


class Graph(nx.Graph):
    def __init__(self, target=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.laplacians = None
        self.v_plus = None

    def get_edge_index(self):
        adj = torch.Tensor(nx.to_numpy_array(self))
        edge_index, _ = dense_to_sparse(adj)
        return edge_index

    def get_edge_attr(self):
        features = []
        for _, _, edge_attrs in self.edges(data=True):
            data = []

            if edge_attrs["label"] is not None:
                data.extend(edge_attrs["label"])

            if edge_attrs["attrs"] is not None:
                data.extend(edge_attrs["attrs"])

            features.append(data)
        return torch.Tensor(features)

    def get_x(self, use_node_attrs=False, use_node_degree=False, use_one=False, use_shared=False, 
              use_1hot=False, use_random_normal=False, use_pagerank=False, use_eigen=False, 
              use_deepwalk=False):
        """
        only load node attrs ! 2023.01.30, lzd
        """
        
        features = []


        for node, node_attrs in self.nodes(data=True):
            data = []
            if node_attrs["label"] is not None:
                data.extend(node_attrs["label"])
            
            if use_node_attrs and node_attrs["attrs"] is not None:
                data.extend(node_attrs["attrs"])

            # if use_node_degree:
            #     data.extend([self.degree(node)])

            # if use_one:
            #     data.extend([1])
            
            # if use_shared:
            #     data.extend([1] * 50)
            
            # if use_1hot:
            #     arr = Graph_whole_embedding(torch.LongTensor([node-1]))
            #     data.extend(list(arr.view(-1).detach().numpy()))

            # if use_random_normal:
            #     arr = Graph_whole_embedding[node-1, :]
            #     data.extend(list(arr))

            # if use_pagerank:
            #     data.extend([Graph_whole_pagerank[node]] * 50)

            # if use_eigen:
            #     data.extend(list(Graph_whole_eigen[node-1]))

            # if use_deepwalk:
            #     data.extend(list(Graph_whole_deepwalk[node-1]))

            features.append(data)

            # print(len(data))
            # exit()
            feas = torch.Tensor(features)
        return feas

    def get_target(self, classification=True):
        if classification:
            return torch.LongTensor([self.target])

        return torch.Tensor([self.target])

    @property
    def has_edge_attrs(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["attrs"] is not None

    @property
    def has_edge_labels(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["label"] is not None

    @property
    def has_node_attrs(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["attrs"] is not None

    @property
    def has_node_labels(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["label"] is not None
