import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

import torch
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, fea_dim, target_dim, config):
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        self.aggregation = config['aggregation']  # can be mean or max
        self.node_pool = config['node_pool']
        print('node_pool: ', self.node_pool)

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = fea_dim if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, target_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':# ???? why need that? overfitting if True.
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        if self.node_pool == 'max':
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
