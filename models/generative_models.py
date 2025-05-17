import sys,os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torch.nn import ReLU, Sigmoid, Tanh
import torch.nn.functional as F
# import from GIN
from torch_geometric.nn import GCNConv

import numpy as np

from models.graph_classifiers.GIN import GIN
from utils.utils import rewire_edge_index, perturb_batch_edge_index, dense_to_edge_index, edge_index_to_dense
from models.vgae_models import GraphVGAE, GaussianGCNEncoder, DenseGCNConv, ORIVGAE, GraphConvolution, NodeVGAE
from models.vgae_models import PiorEncoder, HGG, GraphPriorEncoder
from node_task.utils import normalize_adj
import argparse

# import dense_to_sparse:
from torch_geometric.utils import dense_to_sparse


def dense_adj_to_edge_index(dense_adj_batch, batch):
    edge_indices = []

    # Start index for the node numbering in each graph
    start_idx = 0

    for i in range(batch.max().item() + 1):
        # Get the dense adjacency matrix for the current graph
        dense_adj = dense_adj_batch[i]
        num_nodes = dense_adj.size(0)

        # Find the indices of non-zero elements (edges) in the adjacency matrix
        # The returned indices will be in a 2xN format, where N is the number of edges
        edges = dense_adj.nonzero(as_tuple=False).t().contiguous()

        # Adjust these indices to account for the overall node indexing
        edges += start_idx

        # Append to the global edge index list
        edge_indices.append(edges)

        # Update the start index for the next graph
        start_idx += num_nodes

    # Concatenate all edge indices to form the global edge index tensor
    global_edge_index = torch.cat(edge_indices, dim=1)

    return global_edge_index




class CompactPooling(nn.Module):
    def __init__(self, args, K, N):
        super(CompactPooling, self).__init__()
        self.CompM = nn.Parameter(torch.Tensor(K, N).cuda())
        nn.init.normal_(self.CompM, mean=0.01, std=0.01)

    def forward(self, x):
        x = torch.matmul(self.CompM, x)
        x = torch.sum(x, dim=-2).squeeze()
        return x
    

class GateGraphPooling(nn.Module):
    def __init__(self, args, N):
        super(GateGraphPooling, self).__init__()
        self.args = args
        self.N = N
        self.gate =nn.Parameter(torch.FloatTensor(self.N))
        self.reset_parameters()
        
        
    def forward(self, x):
        """ignore the following dimensions after the 3rd one.
        Args:
            x (tensor): shape: B,N,C,...
        Returns:
            x shape: B,C,...
        """
        shape = x.shape
        if len(shape) > 3:
            x = torch.einsum('btnc, n -> btc', x, self.gate)
        else:
            x = torch.einsum('bnc, n -> bc', x, self.gate)
        return x
    
    def reset_parameters(self):
        nn.init.normal_(self.gate, mean=0.01, std=0.01)


class CatPooling(nn.Module):
    def __init__(self):
        super(CatPooling, self).__init__()
        pass

    def forward(self, x):
        # input B*NC -> B*C
        return torch.flatten(x, start_dim=-2)


class AttGraphPooling(nn.Module):
    def __init__(self, args, N, in_dim, hid_dim):
        super(AttGraphPooling, self).__init__()
        self.args = args
        self.N = N
        self.Q = nn.Linear(in_dim, hid_dim)
        self.K = nn.Linear(in_dim, hid_dim)
        self.V = nn.Linear(in_dim, hid_dim)
        if args.agg_type == 'gate':
            self.gate =nn.Parameter(torch.FloatTensor(self.N))
        self.reset_parameters()
        
    def forward(self, x):
        """ignore the following dimensions after the 3rd one.
        Args:
            x (tensor): shape: B,N,C
        Returns:
            x shape: B,C
        """
        x = x.transpose(2, -1)  # make last dimension is channel.
        Q = self.Q(x) # BNC BNC.
        K = self.K(x)
        V = self.V(x)
        
        att = torch.bmm(Q, K.transpose(1,2))/self.N**0.5
        att = torch.softmax(att, dim=1)
        x = torch.bmm(att, V)  # bnc.
        # TODO: add gated? or sum? or linear?, try 3.
        if self.args.agg_type == 'gate':
        # NOTE: method1: gated
            x = torch.einsum('bnc, n -> bc', x, self.gate)
        # NOTE: method2:cat
        elif self.args.agg_type == 'cat':
        # NOTE: method3:sum
            x = torch.flatten(x, start_dim=1)
        elif self.args.agg_type == 'sum':
            x = torch.sum(x, dim=1)
        
        return x
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.Q.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.K.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.V.weight, mode='fan_in')
        if self.args.agg_type == 'gate':
            nn.init.normal_(self.gate, mean=0.01, std=0.01)


class PSampling(nn.Module):
    def __init__(self, fea_dim, edge_attr_dim, config, dim_z=128):
        super().__init__()
        self.dim_z = dim_z
        self.gnn_mean =  GIN(fea_dim, edge_attr_dim, self.dim_z, config, act=Sigmoid)
        self.gnn_std =  GIN(fea_dim, edge_attr_dim, self.dim_z, config, act=Sigmoid)
        # add layer norm on mean and std:
        self.bn_mean = nn.LayerNorm(self.dim_z)
        self.bn_std = nn.LayerNorm(self.dim_z)

        
    def forward(self, data=None, x=None, adj=None):
        mean_v = self.bn_mean(self.gnn_mean(data))
        std_v = self.bn_mean(self.gnn_std(data))
        
        epsilon = torch.randn(size=mean_v.shape, device=mean_v.device)
        samples = mean_v + epsilon * std_v
        
        return samples, mean_v, std_v



class Conv1dGenerator(nn.Module):
    def __init__(self, fea_dim, output_channels=12, component=1, num_layers=3):
        super(Conv1dGenerator, self).__init__()
        self.layers = nn.ModuleList()
        hidden_channels = 256
        
        self.layers.append(nn.Conv1d(1, hidden_channels, kernel_size=1))
        
        for _ in range(num_layers-2):
            self.layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1))
            
        self.layers.append(nn.Conv1d(hidden_channels, output_channels, kernel_size=1))
        
        self.Q_linear = nn.Linear(hidden_channels, 128)
        self.K_linear = nn.Linear(fea_dim, 128)

    def forward(self, Z, X, batch):
        Z = Z.unsqueeze(1)
        X = X.unsqueeze(1)
        for layer in self.layers:
            Z = F.relu(layer(Z))
            X = F.relu(layer(X))
        
        Q = self.Q_linear(Z)
        K = self.K_linear(X)
        
        Q_expand = Q.index_select(0, batch)
        # element-wise product of Q_expand and K
        V = K * Q_expand
        return V


class MeanGenerator(Conv1dGenerator):
    pass


class VarianceGenerator(Conv1dGenerator):
    def forward(self, Z, X, batch):
        V = super().forward(Z, X, batch)
        return F.relu(V)


class CateGenerator(Conv1dGenerator):
    def forward(self, S, X, batch):
        V = super().forward(S, X, batch)
        return F.softmax(V, dim=-1)


class HVOEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, K=1, sparse=False, args=None):
        super(HVOEncoder, self).__init__()
        self.K = K # component number of mixture model
        # TODO: in baseline, we fix prior as constant, ignore it.
        # self.prior_G = GaussianGCNEncoder(args, args.fea_dim, args.hid_dim, args.hid_dim, args.num_layers, K=self.K)
        # use samples drawn from prior_G as the parameters of lambda, where the lambda is the parameters of mixture model.
        self.Z_G = GaussianGCNEncoder(input_dim, hidden_dim1, hidden_dim2, K=self.K, act='relu', sparse=sparse)
    

    def get_logstd(self):
        return self.logstd
    
    def get_mean(self):
        return self.mean


    def sample_z(self, pi_k, mean_k, logvar_k):
        # use categorical pi_k to select the guassian component:
        # pi_k shape: BxNxK, mean_k shape: BxNxKxD, logvar_k shape: BxNxKxD
        # NOTE: use MCMC to sample or use softmax to weighted sum of all gaussian components.
        # sample from categorical distribution:
        if not self.training:
            if self.K == 1:
                return mean_k
            
            mean_k = torch.matmul(mean_k, pi_k.unsqueeze(-1)).squeeze()
            return mean_k
        
        if self.K == 1:
            epsilon = torch.randn_like(logvar_k)
            sample = mean_k + torch.exp(0.5*logvar_k) * epsilon
            return sample

        use_prop = False
        if use_prop:
            k = torch.multinomial(pi_k, num_samples=1)
            mean_k = mean_k.gather(dim=-2, index=k)
            logvar_k = logvar_k.gather(dim=-2, index=k)
            epsilon = torch.randn_like(logvar_k)
            sample = mean_k + torch.exp(0.5*logvar_k) * epsilon
        else:
            # TODO: Gumbel softmax trick:
            mean_k = torch.matmul(mean_k, pi_k.unsqueeze(-1)).squeeze()
            logvar_k = torch.matmul(logvar_k, pi_k.unsqueeze(-1)).squeeze()
            epsilon = torch.randn_like(logvar_k)
            sample = mean_k + torch.exp(0.5*logvar_k) * epsilon
        return sample
    
    def forward(self, x, adj):
        z_mean_k, z_log_std_k, pi_k = self.Z_G(x, adj)
        
        self.mean = z_mean_k
        self.logstd = z_log_std_k
        
        Z = self.sample_z(pi_k, z_mean_k, z_log_std_k)
        return Z


class GSampling(nn.Module):
    def __init__(self, fea_dim, M, K, num_layers):
        super(GSampling, self).__init__()
        self.M = M
        self.K = K # component number of mixture model
        
        self.mean_generator = MeanGenerator(fea_dim, M, K, num_layers)
        self.variance_generator = VarianceGenerator(fea_dim, M, K, num_layers)
        
        if self.K > 1:
            self.cate_generator = CateGenerator(M, K, num_layers)
            self.tau = 0.1


    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mean + eps*std


    def forward(self, Z, X, batch):
        
        mean = self.mean_generator(Z, X, batch)
        variance = self.variance_generator(Z, X,batch)
        PI = self.reparameterize(mean, variance)
        
        if self.K > 1:
            cate_logits = self.cate_generator(Z, X, batch)
            gumbel_softmax = F.gumbel_softmax(cate_logits, dim=-1, tau=self.tau, hard=False)
            PI = PI * gumbel_softmax
        
        # PI shape: torch.Size([2297, 12, 128])
        # generate A_m, m=1,..,12
        # transform PI to adjacency matrix:
        PI = PI.transpose(1, 0)
        P_A = torch.sigmoid(torch.einsum('bnf,bmf->bnm', PI, PI))
        
        P_Am_edge_indexes = dense_to_edge_index(P_A)
        
        
        # sampling from P_A by gumbel softmax trick:
        # A = F.gumbel_softmax(P_A, dim=-1, tau=self.tau, hard=True)
        # only take the upper triangle part for each batch sample:
        # print('A shape:', A.shape)
        # convert adjacency matrix to edge_index
        return P_Am_edge_indexes


def rewire_edge_index(edge_index, ratio):
    """
    Rewire a fraction of the edges in a graph.
    
    Parameters:
    - edge_index (torch.Tensor): The edge index tensor of shape [2, num_edges].
    - ratio (float): The fraction of edges to rewire.
    
    Returns:
    - new_edge_index (torch.Tensor): The rewired edge index tensor.
    """
    # Ensure the ratio is between 0 and 1
    ratio = max(min(ratio, 1), 0)
    
    new_edge_index = edge_index.clone()
    
    # Compute the number of edges to rewire
    total_edges = edge_index.size(1)
    num_edges_to_rewire = int(total_edges * ratio)
    
    # Randomly select edges to rewire
    edges_to_rewire = torch.randperm(total_edges)[:num_edges_to_rewire]
     
    # Get all unique nodes in the graph
    nodes = torch.unique(edge_index)
    
    # Randomly select end nodes for the edges to be rewired
    end_nodes = nodes[torch.randint(nodes.shape[0], (edges_to_rewire.shape[0], ))]
    new_edge_index[1, edges_to_rewire] = end_nodes.flatten()
    
    return new_edge_index



# TODO: HVO:

class GNNPredictor(torch.nn.Module):
    def __init__(self, gnn):
        super(GNNPredictor, self).__init__()
        self.gnn = gnn

    def forward(self, data, x=None, edge_index=None):
        x = self.gnn(data=data, x=x, edge_index=edge_index)
        self.features = x.detach()  # Save the features
        y = F.log_softmax(x, dim=1)
        return y



import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class OriGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, adj=None):
        super(OriGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.save_features = False
        self.adj = adj

    def forward(self, x, adj):
        if self.adj is not None:
            adj = self.adj
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        if self.save_features:
            self.features = x.detach()  # Save the features
            
        return F.log_softmax(x, dim=1)
    
    
    

class DenseGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dims=[16], dropout=0.5):
        super(DenseGCN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(DenseGCNConv(num_features, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.convs.append(DenseGCNConv(hidden_dims[i-1], hidden_dims[i]))
        self.convs.append(DenseGCNConv(hidden_dims[-1], num_classes))
        self.save_features = False
        self.dropout = dropout

    def forward(self, data=None, x=None, edge_index=None):
        x = data.x if x is None else x
        edge_index = data.edge_index if edge_index is None else edge_index

        # Pass through the GCNConv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:  # No activation & dropout on the last layer
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.save_features:
            self.features = x.detach()  # Save the features
        
        return x



class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dims=[16], dropout=0.3):
        super(GCN, self).__init__()
        
        # Create a list of GCNConv layers based on the hidden_dims
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.convs.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        self.convs.append(GCNConv(hidden_dims[-1], num_classes))
        self.save_features = True
        self.dropout = dropout

    def forward(self, data, x=None, edge_index=None):
        x = data.x if x is None else x
        edge_index = data.edge_index if edge_index is None else edge_index

        # Pass through the GCNConv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:  # No activation & dropout on the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.save_features:
            self.features = x.detach()  # Save the features
        
        return x

from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = GATConv(
                nfeat,
                nhid,
                heads=heads,
                dropout=dropout,
                bias=with_bias)

        self.conv2 = GATConv(
                nhid * heads,
                nclass,
                heads=output_heads,
                concat=False,
                dropout=dropout,
                bias=with_bias)

        print('using heads:', heads)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.save_features = True

    def forward(self, data, x=None, edge_index=None):
        x = data.x if x is None else x
        edge_index = data.edge_index if edge_index is None else edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if self.save_features:
            self.features = x.detach()  # Save the features
        
        return x


class GraphDGGN:
    def __init__(self, fea_dim, edge_attr_dim, target_dim, config, adj=None, args=None):
        self.args = args
        self.N = adj.shape[0] if adj is not None else None

        M = 6
        z_dim = 32
        hid_gnn_dim = 32
        
        agreement_dim = 32
        self.target_dim = target_dim
        
        self.M = M
        self.save_features = True
        self.rewire_ratio = config['rewire_ratio']
        args = argparse.Namespace(**config.config) if args is None else args
        self.args = args
        self.gen_type = args.gen_type

        self.ggn_gnn_type = self.args.ggn_gnn_type
        # self.gnn = GIN(fea_dim, edge_attr_dim, hid_gnn_dim, config)
        
        # self.gnn = DenseGCN(fea_dim, self.target_dim, hidden_dims=[hid_gnn_dim, hid_gnn_dim])
        if self.args.ggn_gnn_type == 'gcn':
            if self.gen_type == 'graph_hgg':
                self.gnn = GCN(fea_dim, target_dim, [hid_gnn_dim], dropout=config['dropout'])
            else:
                self.gnn = DenseGCN(fea_dim, target_dim, hidden_dims=[hid_gnn_dim], dropout=config['dropout'])
        elif self.args.ggn_gnn_type == 'gat':
            self.gnn = GAT(fea_dim, hid_gnn_dim, target_dim, heads=args.gat_heads, dropout=config['dropout'], device=args.device)
        elif self.args.ggn_gnn_type == 'gin':
            self.gnn = GIN(fea_dim, edge_attr_dim, hid_gnn_dim, config, act=Sigmoid)
        else:
            raise NotImplementedError('ggn_gnn_type not implemented:', args.ggn_gnn_type)

        # TODO: replace to GAT

        
        self.config = config
        self.gen_type = 'gsample' if 'gen_type' not in config else config['gen_type']
        
        if self.gen_type == 'gsample':
            self.p_sampling = PSampling(fea_dim, edge_attr_dim, config, dim_z=z_dim)
            self.g_sampling = GSampling(fea_dim, M, K=1, num_layers=3)
        elif self.gen_type == 'graph_hgg':
                        # ?? sample prior from mixture gaussian model.
            self.classifier = nn.Sequential(
                nn.Linear(hid_gnn_dim, agreement_dim),
                nn.ReLU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(agreement_dim, target_dim)
            )
            p_sampler = GraphPriorEncoder(args, encoder=HVOEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, sparse=True, args=args))
            g_sampler = HVOEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, K=self.args.k_components, sparse=True, args=args)
            
            self.graph_gen = GraphVGAE(fea_dim, hid_gnn_dim, hid_gnn_dim, p_sampler=p_sampler, g_sampler=g_sampler, args=args)

        elif self.gen_type == 'node_vgae':
            # TODO: mask is attention based matrix parameters, N is node number, k is component number.
            if bool(self.args.use_hvo):
                print('use hov:', bool(self.args.use_hvo))
                encoder = HVOEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, K=self.args.k_components)
            else:
                encoder = GaussianGCNEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, config=config)
            
            self.graph_gen = NodeVGAE(fea_dim, hid_gnn_dim, hid_gnn_dim//2, encoder=encoder)
            # self.graph_gen = ORIVGAE(fea_dim, hid_gnn_dim, hid_gnn_dim)
        elif self.gen_type == 'node_hgg':
            # ?? sample prior from mixture gaussian model.
            p_sampler = PiorEncoder(self.N, K=hid_gnn_dim//2)
            g_sampler = HVOEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, K=self.args.k_components)
            self.graph_gen = HGG(p_sampler=p_sampler, g_sampler=g_sampler, adj=None)

        self.adj = adj
        self.epoch = 0
        self.weight_tensor = self.cal_ce_weight(adj)


    def state_dict(self):
        state_dict = {}
        state_dict['gnn'] = self.gnn.state_dict()
        state_dict['graph_gen'] = self.graph_gen.state_dict()
        return state_dict
    

    def load_state_dict(self, state_dict):
        self.gnn.load_state_dict(state_dict['gnn'])
        self.graph_gen.load_state_dict(state_dict['graph_gen'])
        return self
    

    def to(self, device):
        self.gnn.to(device)
        self.graph_gen.to(device)
        self.device = device
        self.classifier = self.classifier.to(device)
        return self


    def modify_aug_A(self, aug_dense_As, ori_As, batch):
        
        # print device of all:
        new_adjs = []
        for i, aug_A in enumerate(aug_dense_As):
            ori_adj = ori_As[i]
            aug_A = torch.where(aug_A < 0.99, torch.zeros_like(aug_A), aug_A)
            aug_A = torch.where(aug_A >= 0.99, torch.ones_like(aug_A), aug_A)

            if self.ggn_gnn_type == 'gcn':
                aug_A = normalize_adj(aug_A)
            # new_adj = ori_mask * ori_adj + (1-ori_mask) * aug_A
            # aug_A = normalize_adj(aug_A)
            # combine edge_index instead of dense adj:
            new_adjs.append(self.args.ori_ratio * ori_adj + self.args.aug_ratio * aug_A)

        new_edge_index = dense_adj_to_edge_index(new_adjs, batch)

        return new_edge_index


    def sce_loss(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss
    
    def update_prior_graph(self, aug_dense_As, ori_As):
        loss = self.graph_gen.loss_prior(aug_dense_As, ori_As)
        return loss

        
    def cal_ce_weight(self, adj):
        if adj is None:
            return None
        
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        weight_mask = adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(adj.device)
        weight_tensor[weight_mask] = pos_weight


    def _loss_re(self, aug_A, ori_A):
        norm = ori_A.shape[0] * ori_A.shape[0] / float((ori_A.shape[0] * ori_A.shape[0] - ori_A.sum()) * 2)
        if self.weight_tensor is None:
            weight_tensor = self.cal_ce_weight(ori_A)
        else:
            weight_tensor = self.weight_tensor

        recons_loss = log_lik = norm * F.binary_cross_entropy(aug_A.view(-1), ori_A.view(-1), weight = weight_tensor)
        
        loss = recons_loss
        logstd = self.graph_gen.get_logstd()
        mean = self.graph_gen.get_mean()
        kl_divergence = 0.5/ aug_A.size(0) * (1 + 2*logstd - mean**2 - torch.exp(logstd)**2).sum(1).mean()
        loss -= kl_divergence
        return loss
 
    def loss_re(self, aug_dense_As, ori_As):
        
        loss = 0
        for i, aug_A in enumerate(aug_dense_As):
            ori_adj = ori_As[i]
            loss += self._loss_re(aug_A, ori_adj)

        return loss

    def eval(self):
        self.graph_gen.eval()
        self.gnn.eval()
        

    def generate_graph(self, data, x_ori, edge_index):
        aug_edge_index, dense_A_preds = self.graph_gen(x=x_ori, adj=edge_index, batch=data.batch)
        self.aug_edge_index = aug_edge_index
        self.dense_A_preds = dense_A_preds
        return aug_edge_index, dense_A_preds
    
    def before_forward(self, epoch):
        self.epoch = epoch


    def graph_predict(self, data=None, x=None, edge_index=None, batch=None):
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gnn(data, x=x, edge_index=edge_index)
        if self.save_features:
            self.features = x.detach()
            
        # y = self.classifier(x)
        y = x

        outputs= {'y': y,
                  'target_edge_index': edge_index,
                  'batch': data.batch}
        
        return outputs


    def graph_forward(self, data=None, x=None, edge_index=None, batch=None):
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        pert_op='drop'
        
        if self.gen_type == 'gsample':
            Z,  mean_v, std_v = self.p_sampling(data) # TODO
            Am = self.g_sampling(Z, x, batch)
        elif self.gen_type == 'mock':
            pos_edge_index = [perturb_batch_edge_index(data, ratio=0.02, op=pert_op),
                              perturb_batch_edge_index(data, ratio=0.03, op=pert_op)]
            Am = pos_edge_index
        elif self.gen_type == 'vgae':
            # pretrain ????? 
            # NOTE: maybe we could extend VGAE to prior ??? use perturbed edge_index to self-train ?
            Am, Am_dense_adjs = [], []
            for _ in range(2):
                edge_index_preds, dense_adj_preds = self.vgae(x, adj=edge_index, batch=batch)
                Am.append(edge_index_preds)
                Am_dense_adjs.append(dense_adj_preds)
            # Am is not sparse, so do not use edge_index.
            mean_v, std_v = self.vgae.mean, self.vgae.logstd
            # check whether mean_v is nan:
            if torch.isnan(mean_v).any():
                print('mean_v is nan')
                raise ValueError
        else:
            raise NotImplementedError
        
        pos_edge_index = [perturb_batch_edge_index(data, ratio=0.02, op=pert_op), 
                        perturb_batch_edge_index(data, ratio=0.03, op=pert_op)]
        
        neg_edge_index = [perturb_batch_edge_index(data, ratio=0.3, op=pert_op),
                        perturb_batch_edge_index(data, ratio=0.4, op=pert_op)]

        H_A0 = self.gnn(data=data)
        Am_agre, pos_agre, neg_agre = None, None, None
    
        if self.training and self.epoch > -11:
            g_h = H_A0
            if self.epoch % 5 == 0:
                print('epoch of model:', self.epoch)
        else:
            # self.gnn.eval()
            # ignore:
            H_Am = [self.gnn(data=data, edge_index=Am[i]) for i in range(len(Am))]
            
            g_h = torch.stack(H_Am, dim=0).sum(dim=0) + H_A0

            Am_agre = [self.agreement_encoder(h) for h in H_Am]
            
            g_h_pos = [self.gnn(data=data, edge_index=edges) for edges in pos_edge_index]
            g_h_pos.append(H_A0)
            
            # g_h_neg = [self.gnn(data=data, edge_index=edges) for edges in neg_edge_index]
            
            pos_agre = [self.agreement_encoder(h) for h in g_h_pos]
            # neg_agre = [self.agreement_encoder(h) for h in g_h_neg]
            
            neg_agre = None
            
        if not self.training:
            print('evaluation mode')
        
        if self.training: # if testing, do not update the parameters of gnn
            self.gnn.train()
                
        # TODO: VGAE should add match loss.
            
        y = self.classifier(g_h)
        
        # design loss function:
        # make output as dict:
        outputs= {'y': y,
                  'Am_agre': Am_agre,
                  'pos_agre': pos_agre,
                  'neg_agre': neg_agre,
                  'mean_v': mean_v,
                  'std_v': std_v,
                  'Am': Am,
                  'Am_dense_adjs': Am_dense_adjs,
                  'target_edge_index': data.edge_index,
                  'batch': data.batch}
        
        return outputs




class DGGN:
    def __init__(self, fea_dim, edge_attr_dim, target_dim, config, adj=None, args=None):
        self.args = args
        self.N = adj.shape[0] if adj is not None else None

        M = 6
        z_dim = 32
        hid_gnn_dim = 32
        
        agreement_dim = 32
        self.target_dim = target_dim
        
        self.M = M
        self.save_features = True
        self.rewire_ratio = config['rewire_ratio']
        args = argparse.Namespace(**config.config) if args is None else args
        self.args = args
        self.gen_type = args.gen_type

        self.ggn_gnn_type = self.args.ggn_gnn_type
        # self.gnn = GIN(fea_dim, edge_attr_dim, hid_gnn_dim, config)
        
        # self.gnn = DenseGCN(fea_dim, self.target_dim, hidden_dims=[hid_gnn_dim, hid_gnn_dim])
        if self.args.ggn_gnn_type == 'gcn':
            if self.gen_type == 'graph_hgg':
                self.gnn = GCN(fea_dim, target_dim, [hid_gnn_dim], dropout=config['dropout'])
            else:
                self.gnn = DenseGCN(fea_dim, target_dim, hidden_dims=[hid_gnn_dim], dropout=config['dropout'])
        elif self.args.ggn_gnn_type == 'gat':
            self.gnn = GAT(fea_dim, hid_gnn_dim, target_dim, heads=args.gat_heads, dropout=config['dropout'], device=args.device)
        elif self.args.ggn_gnn_type == 'gin':
            self.gnn = GIN(fea_dim, edge_attr_dim, hid_gnn_dim, config, act=Sigmoid)
        else:
            raise NotImplementedError('ggn_gnn_type not implemented:', args.ggn_gnn_type)

        # TODO: replace to GAT
        self.graph_gen = None
        
        self.config = config
        self.gen_type = 'gsample' if 'gen_type' not in config else config['gen_type']
        
        if self.gen_type == 'gsample':
            self.p_sampling = PSampling(fea_dim, edge_attr_dim, config, dim_z=z_dim)
            self.g_sampling = GSampling(fea_dim, M, K=1, num_layers=3)
        elif self.gen_type == 'graph_hgg':
                        # ?? sample prior from mixture gaussian model.
            # self.classifier = nn.Sequential(
            #     nn.Linear(hid_gnn_dim, agreement_dim),
            #     nn.ReLU(),
            #     nn.Dropout(self.args.dropout),
            #     nn.Linear(agreement_dim, target_dim)
            # )
            p_sampler = GraphPriorEncoder(args, encoder=HVOEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, sparse=True))
            g_sampler = HVOEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, K=self.args.k_components, sparse=True)
            
            self.graph_gen = GraphVGAE(fea_dim, hid_gnn_dim, hid_gnn_dim, p_sampler=p_sampler, g_sampler=g_sampler, args=args)

        elif self.gen_type == 'node_vgae':
            # TODO: mask is attention based matrix parameters, N is node number, k is component number.
            if bool(self.args.use_hvo):
                print('use hov:', bool(self.args.use_hvo))
                encoder = HVOEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, K=self.args.k_components)
            else:
                encoder = GaussianGCNEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, config=config)
            
            self.graph_gen = NodeVGAE(fea_dim, hid_gnn_dim, hid_gnn_dim//2, encoder=encoder)
            # self.graph_gen = ORIVGAE(fea_dim, hid_gnn_dim, hid_gnn_dim)
        elif self.gen_type == 'node_hgg':
            # ?? sample prior from mixture gaussian model.
            p_sampler = PiorEncoder(self.N, K=hid_gnn_dim//2)
            g_sampler = HVOEncoder(fea_dim, hid_gnn_dim, hid_gnn_dim//2, K=self.args.k_components, args=args)
            self.graph_gen = HGG(p_sampler=p_sampler, g_sampler=g_sampler, adj=None)

        self.adj = adj
        self.epoch = 0
        self.weight_tensor = self.cal_ce_weight(adj)




    def state_dict(self):
        state_dict = {}
        state_dict['gnn'] = self.gnn.state_dict()
        state_dict['graph_gen'] = self.graph_gen.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.gnn.load_state_dict(state_dict['gnn'])
        self.graph_gen.load_state_dict(state_dict['graph_gen'])
        return self
    
    def to(self, device):
        self.gnn.to(device)
        if self.graph_gen is not None:
            self.graph_gen.to(device)
        self.device = device
        return self


    def modify_aug_A(self, aug_A, ori_adj):
        
        # Set the diagonal of aug_A to be 1
        # mask = torch.eye(aug_A.shape[0], device=aug_A.device).bool()
        # ones = torch.ones_like(aug_A)
        # aug_A = torch.where(mask, ones, aug_A)

        # print device of all:
        aug_A = torch.where(aug_A < 0.99, torch.zeros_like(aug_A), aug_A)
        aug_A = torch.where(aug_A >= 0.99, torch.ones_like(aug_A), aug_A)

        if self.ggn_gnn_type == 'gcn':
            aug_A = normalize_adj(aug_A)
        # new_adj = ori_mask * ori_adj + (1-ori_mask) * aug_A
        # aug_A = normalize_adj(aug_A)

        new_adj = self.args.ori_ratio * ori_adj + self.args.aug_ratio * aug_A

        return new_adj


    def sce_loss(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss
    

    def update_prior(self, data):
        x, edge_indx = data.x, data.edge_index
        l1_reg, l2_reg, feature_reg = self.args.l1_reg, self.args.l2_reg, self.args.feature_reg
        return self.graph_gen.loss_prior(x, l1_reg, l2_reg, feature_reg)
        
    def cal_ce_weight(self, adj):
        if adj is None:
            return None
        
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        weight_mask = adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(adj.device)
        weight_tensor[weight_mask] = pos_weight

    def loss_re(self, aug_A):
        # Create Model
        ori_A = self.adj
        
        norm = ori_A.shape[0] * ori_A.shape[0] / float((ori_A.shape[0] * ori_A.shape[0] - ori_A.sum()) * 2)
        if self.weight_tensor is None:
            weight_tensor = self.cal_ce_weight(ori_A)
        else:
            weight_tensor = self.weight_tensor

        recons_loss = log_lik = norm * F.binary_cross_entropy(aug_A.view(-1), ori_A.view(-1), weight = weight_tensor)
        
        # TODO: add sparsity loss:

        sparse_loss = torch.norm(aug_A, p=1) # too large
        
        # # TODO: add max distance vector:
        # ball_loss = self.hyper_ball_loss(self.graph_gen.encoder.mean)
        
        # # print each loss in one line:
        # print(f'recons_loss: {recons_loss.item()}, sparse_loss: {sparse_loss.item()}, ball_loss: {ball_loss.item()}')
        
        # loss = recons_loss + 5e-4 * sparse_loss + ball_loss
        # loss = recons_loss +  5e-6 * sparse_loss
        loss = recons_loss
        logstd = self.graph_gen.get_logstd()
        mean = self.graph_gen.get_mean()
        kl_divergence = 0.5/ aug_A.size(0) * (1 + 2*logstd - mean**2 - torch.exp(logstd)**2).sum(1).mean()
        loss -= kl_divergence
        
        # loss_symmetric = torch.norm(aug_A - aug_A.t(), p="fro")

        # loss_l1 = torch.norm(aug_A, 1)
        # loss_fro = torch.norm(aug_A - ori_A, p='fro')
        # print('loss_fro:', loss_fro.item())
        # print('loss kl:', loss.item(), f' std: {self.graph_gen.encoder.logstd}, mean: {self.graph_gen.encoder.mean}')
        # loss += loss_fro
        
        return loss

    def eval(self):
        self.graph_gen.eval()
        self.gnn.eval()
        
    def generate_graph(self, data, x_ori, edge_index):
        aug_A = self.graph_gen(x=x_ori, adj=edge_index)
        self.aug_A = aug_A
        return aug_A
    
    def before_forward(self, epoch):
        self.epoch = epoch
    
    def predict(self, data, x=None, edge_index=None, batch=None):
        if self.ggn_gnn_type == 'gat':
            # trans edge_index to sparse 2xE matrix from dense NxN matrix:
            edge_index, _ = dense_to_sparse(edge_index)

        x = self.gnn(data, x=x, edge_index=edge_index)
        if self.save_features:
            self.features = x.detach()  # Save the features
        y = F.log_softmax(x, dim=1)
        return y

    def graph_predict(self, data=None, x=None, edge_index=None, batch=None):
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.ggn_gnn_type == 'gat':
            # trans edge_index to sparse 2xE matrix from dense NxN matrix:
            edge_index, _ = dense_to_sparse(edge_index)
        
        x = self.gnn(data, x=x, edge_index=edge_index)
        if self.save_features:
            self.features = x.detach()
            
        # y = self.classifier(x)
        y = x
        # design loss function:
        # make output as dict:
        outputs= {'y': y,
                  'target_edge_index': edge_index,
                  'batch': data.batch}
        print('graph predict')
        return outputs


    def graph_forward(self, data=None, x=None, edge_index=None, batch=None):
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        pert_op='drop'
        
        if self.gen_type == 'gsample':
            Z,  mean_v, std_v = self.p_sampling(data) # TODO
            Am = self.g_sampling(Z, x, batch)
        elif self.gen_type == 'mock':
            pos_edge_index = [perturb_batch_edge_index(data, ratio=0.02, op=pert_op),
                              perturb_batch_edge_index(data, ratio=0.03, op=pert_op)]
            Am = pos_edge_index
        elif self.gen_type == 'vgae':
            # pretrain ????? 
            # NOTE: maybe we could extend VGAE to prior ??? use perturbed edge_index to self-train ?
            Am, Am_dense_adjs = [], []
            for _ in range(2):
                edge_index_preds, dense_adj_preds = self.vgae(x, adj=edge_index, batch=batch)
                Am.append(edge_index_preds)
                Am_dense_adjs.append(dense_adj_preds)
            # Am is not sparse, so do not use edge_index.
            mean_v, std_v = self.vgae.mean, self.vgae.logstd
            # check whether mean_v is nan:
            if torch.isnan(mean_v).any():
                print('mean_v is nan')
                raise ValueError
        else:
            raise NotImplementedError
        
        pos_edge_index = [perturb_batch_edge_index(data, ratio=0.02, op=pert_op), 
                        perturb_batch_edge_index(data, ratio=0.03, op=pert_op)]
        
        neg_edge_index = [perturb_batch_edge_index(data, ratio=0.3, op=pert_op),
                        perturb_batch_edge_index(data, ratio=0.4, op=pert_op)]

        H_A0 = self.gnn(data=data)
        Am_agre, pos_agre, neg_agre = None, None, None
    
        if self.training and self.epoch > -11:
            g_h = H_A0
            if self.epoch % 5 == 0:
                print('epoch of model:', self.epoch)
        else:
            # self.gnn.eval()
            # ignore:
            H_Am = [self.gnn(data=data, edge_index=Am[i]) for i in range(len(Am))]
            
            g_h = torch.stack(H_Am, dim=0).sum(dim=0) + H_A0

            Am_agre = [self.agreement_encoder(h) for h in H_Am]
            
            g_h_pos = [self.gnn(data=data, edge_index=edges) for edges in pos_edge_index]
            g_h_pos.append(H_A0)
            
            # g_h_neg = [self.gnn(data=data, edge_index=edges) for edges in neg_edge_index]
            
            pos_agre = [self.agreement_encoder(h) for h in g_h_pos]
            # neg_agre = [self.agreement_encoder(h) for h in g_h_neg]
            
            neg_agre = None
            
        if not self.training:
            print('evaluation mode')
        
        if self.training: # if testing, do not update the parameters of gnn
            self.gnn.train()
                
        # TODO: VGAE should add match loss.
            
        y = self.classifier(g_h)
        
        # design loss function:
        # make output as dict:
        outputs= {'y': y,
                  'Am_agre': Am_agre,
                  'pos_agre': pos_agre,
                  'neg_agre': neg_agre,
                  'mean_v': mean_v,
                  'std_v': std_v,
                  'Am': Am,
                  'Am_dense_adjs': Am_dense_adjs,
                  'target_edge_index': data.edge_index,
                  'batch': data.batch}
        
        return outputs


# Use this module like this

if __name__ == '__main__':
    
    num_samples = 1000
    dim1 = 3
    dim2 = 3
    input_channels = dim1 * dim2
    output_channels = 30
    num_layers = 3


    X = torch.randn(100, input_channels)  # random tensor for X

    g_sampling = GSampling(output_channels, num_layers)
    PI = g_sampling(X)

    print(PI)
