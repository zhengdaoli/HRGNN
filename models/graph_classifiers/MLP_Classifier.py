import torch
import torch.nn.functional as F
from torch.nn import Linear


class MLPClassifier(torch.nn.Module):

    def __init__(self, fea_dim, target_dim, config):
        super(MLPClassifier, self).__init__()

        hidden_units = config['hidden_units']

        self.fc_global = Linear(fea_dim, hidden_units)
        self.out = Linear(hidden_units, target_dim)

    def forward(self, x, batch):
        return self.out(F.relu(self.fc_global(x)))
