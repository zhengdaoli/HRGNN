import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_add_pool


class DeepMultisets(torch.nn.Module):

    def __init__(self, fea_dim, target_dim, config):
        super(DeepMultisets, self).__init__()

        hidden_units = config['hidden_units']

        self.fc_vertex = Linear(fea_dim, hidden_units)
        self.fc_global1 = Linear(hidden_units, hidden_units)
        self.fc_global2 = Linear(hidden_units, target_dim)

    def forward(self, data):
        x, batch = data.x, data.batch

        x = F.relu(self.fc_vertex(x))
        x = global_add_pool(x, batch)  # sums all vertex embeddings belonging to the same graph!
        x = F.relu(self.fc_global1(x))
        x = self.fc_global2(x)
        return x

