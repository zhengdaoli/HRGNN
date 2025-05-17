import torch
from torch.nn import ReLU
from torch import dropout, nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from models.graph_classifiers.GIN import MyAtomEncoder


class MolecularGraphMLP(torch.nn.Module):

    def __init__(self, fea_dim, edge_attr_dim, target_dim, config):
        super(MolecularGraphMLP, self).__init__()
        hidden_dim = config['hidden_units']
        dropout = config['dropout'] if 'dropout' in config else 0.4
        print('fea_dim: ', fea_dim)
        print('hidden_dim: ', hidden_dim)
        print('target_dim: ', target_dim)
        self.target_dim = target_dim
        print('dropout:', dropout)
        self.act_func = nn.ReLU
        if 'activation' in config and config['activation'] == 'sigmoid':
            self.act_func = nn.Sigmoid
        self.bn1 = nn.BatchNorm1d(fea_dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(fea_dim, hidden_dim), self.act_func(),
                                       nn.Dropout(dropout),
                                    #    torch.nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
                                    #    nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_dim, target_dim),  self.act_func())

    def forward(self, data):
        # TODO: use graph-wise feature: g_x 
        if 'g_x' in data:
            # print('using g_x:', data['g_x'][:20],data['g_x'][20:-1] )
            if data['g_x'].dim() == 1:
                h_g = data['g_x'].unsqueeze(dim=1)
            else:
                h_g = data['g_x']
            
            if h_g.shape[0] > 1:
                h_g = self.bn1(h_g)
            result = self.mlp(h_g)
            
            # print('result: ', result)
            return result
                
        return self.mlp(global_add_pool(data.x, data.batch))
    


class MolecularFingerprint(torch.nn.Module):

    def __init__(self, fea_dim, edge_attr_dim, target_dim, config):
        super(MolecularFingerprint, self).__init__()
        hidden_dim = config['hidden_units']
        print('finger fea_dim:', fea_dim)
        self.target_dim = target_dim
        self.mlp = nn.Sequential(nn.BatchNorm1d(fea_dim),
                                nn.Linear(fea_dim, hidden_dim), ReLU(),
                                nn.Dropout(config['dropout']),
                                # nn.Linear(hidden_dim, hidden_dim), ReLU(),
                                # nn.BatchNorm1d(hidden_dim),
                                nn.Linear(hidden_dim, target_dim), ReLU())
        
    def forward(self, data):
        # return self.mlp(global_mean_pool(data.x.float(), data.batch))
        h = global_add_pool(data.x.float(), data.batch)
        return self.mlp(h)


class AtomMLP(torch.nn.Module):
    def __init__(self, fea_dim, target_dim, config):
        super(AtomMLP, self).__init__()

        self.config = config
        self.dropout = config['dropout']
        self.embeddings_dim = config['hidden_dim']
        hidden_dim = self.embeddings_dim
        
        self.node_encoder = MyAtomEncoder(self.embeddings_dim)
        # self.ln = nn.Linear(fea_dim, self.embeddings_dim)
        self.mol = True
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlp = nn.Sequential(nn.BatchNorm1d(hidden_dim),
                                nn.Linear(hidden_dim, hidden_dim), ReLU(),
                                nn.Dropout(config['dropout']),
                                nn.Linear(hidden_dim, hidden_dim), ReLU())
        
        self.out = nn.Linear(hidden_dim, target_dim)
        self.reset_parameters()

    def reset_parameters(self):
        if self.mol:
            for emb in self.node_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            nn.init.xavier_uniform_(self.node_encoder.weight.data)


    def forward(self, data=None, x=None, edge_index=None, edge_attr=None, batch=None):
        if data is not None:
            x, _, _, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.node_encoder(x)
        h = global_add_pool(h, batch=batch)
        h = self.out(self.mlp(h))

        return h
    