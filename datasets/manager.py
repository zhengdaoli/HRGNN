import my_utils
from dataset_utils import node_feature_utils
from datasets.tu_utils import parse_tu_data, create_graph_from_tu_data, get_dataset_node_num, create_graph_from_nx
from datasets.sampler import RandomSampler, ImbalancedDatasetSampler
from datasets.dataset import GraphDataset, GraphDatasetSubset
from datasets.dataloader import DataLoader
from datasets.data import Data as MyData
from datasets.synthetic_dataset_generator import *
from models.modules import *
from utils.encode_utils import NumpyEncoder
from utils.utils import rewire_edge_index, dropping_edge, adding_edge
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.nn import functional as F
import torch
import numpy as np
from collections import defaultdict
import pickle as pk
from numpy import linalg as LA
import torch.nn as nn
from networkx import normalized_laplacian_matrix
import networkx as nx
from pathlib import Path
import zipfile
import requests
import json
import os
import io
import sys
from copy import deepcopy

import os

from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset
from torch_geometric.datasets import PPI as PPIDataset
from torch_geometric.datasets import QM9, MoleculeNet

from torch_geometric.datasets import Planetoid




from torch_geometric.nn import MessagePassing

from torch_geometric.data import DataLoader as torch_DataLoader
import torch_geometric.utils as torch_utils

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

sys.path.append(os.getcwd())

# import k_gnn

class EdgeAggregator(MessagePassing):
    def __init__(self):
        super(EdgeAggregator, self).__init__(aggr = "add")

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j=0, edge_attr=0):
        # x_j == 0
        return x_j + edge_attr

    def update(self, aggr_out):
        return aggr_out
    
edge_aggregator = EdgeAggregator()

def add_zeros(data):
    data.x = torch.zeros((data.num_nodes, 1), dtype=torch.long)
    return data

# import k_gnn
def add_edge_attr(data):
    data = add_zeros(data)
    data.x = edge_aggregator(data.x, data.edge_index, data.edge_attr)
    return data

def is_pyg_dataset(d_name:str):
    return d_name.startswith('ogb') or d_name.startswith('syn')

def perturb_edge_index(num_nodes, edge_index, ratio, op):
    "op: rewire, drop, add"
    if op == 'rewire':
        return rewire_edge_index(num_nodes, edge_index, ratio)
    elif op == 'drop':
        return dropping_edge(edge_index, ratio)
    elif op == 'add':
        return adding_edge(num_nodes, edge_index, ratio)
    else:
        raise NotImplementedError(f'op: {op} is not implemented')
    

class EmptyNodeFeatureException(Exception):
    def __init__(self) -> None:
        super().__init__("There is no node feature as input!!!")

class GraphDatasetManager:
    def __init__(self, kfold_class=StratifiedKFold, outer_k=10, inner_k=None, seed=42, holdout_test_size=0.1,
                 use_node_degree=False, use_node_attrs=False, use_one=False, use_shared=False, use_1hot=False,
                 use_random_normal=False, use_pagerank=False, use_eigen=False, use_eigen_norm=False,
                 use_deepwalk=False, precompute_kron_indices=False, additional_features: str = None, additional_graph_features: str = None,
                 max_reductions=10, DATA_DIR='./DATA', config={}):

        self.root_dir = Path(DATA_DIR) / self.name
        self.kfold_class = kfold_class
        self.holdout_test_size = holdout_test_size
        self.config = config
        if additional_features is not None:
            add_features = additional_features.strip().split(',')
            self.use_1hot = True if 'use_onehot' in add_features else use_1hot
            self.use_random_normal = True if 'use_random' in add_features else use_random_normal
            self.use_pagerank = True if 'use_pagerank' in add_features else use_pagerank
            self.use_eigen = True if 'use_eigen' in add_features else use_eigen
            self.use_deepwalk = True if 'use_deepwalk' in add_features else use_deepwalk
        else:
            self.use_1hot = use_1hot
            self.use_random_normal = use_random_normal
            self.use_pagerank = use_pagerank
            self.use_eigen = use_eigen
            self.use_deepwalk = use_deepwalk

        self.use_node_degree = use_node_degree
        self.use_node_attrs = use_node_attrs
        self.use_one = use_one
        self.use_shared = use_shared
        self.use_eigen_norm = use_eigen_norm
        self.precompute_kron_indices = precompute_kron_indices
        # will compute indices for 10 pooling layers --> approximately 1000 nodes
        self.KRON_REDUCTIONS = max_reductions
        # 2022.10.02
        self.additional_features = additional_features
        # 2022.10.20
        self.additional_graph_features = additional_graph_features

        self.Graph_whole = None
        self.Graph_whole_pagerank = None
        self.Graph_whole_eigen = None
        
        self.config = config
        self.adjs = None

        self.outer_k = outer_k
        assert (outer_k is not None and outer_k > 0) or outer_k is None

        self.inner_k = inner_k
        assert (inner_k is not None and inner_k > 0) or inner_k is None

        self.seed = seed

        self.raw_dir = self.root_dir / "raw"
        if not self.raw_dir.exists():
            os.makedirs(self.raw_dir)
            self._download()

        self.processed_dir = self.root_dir / "processed"
        print('processed_dir: ', self.processed_dir)
        if not (self.processed_dir / f"{self.name}.pt").exists():
            if not self.processed_dir.exists():
                os.makedirs(self.processed_dir)
            self._process()

        self.dataset = self.load_dataset()
                
        print('!!!! _target_dim: ', self._target_dim)
        print('dataset len: ', len(self.dataset))
        
        use_10_fold = False
        
        if 'use_10_fold' in self.config:
            use_10_fold = self.config['use_10_fold']
            
        use_10_fold = use_10_fold and not self.config['repeat']
        
        if use_10_fold:
            # create new splits
            self.splits_idx = None
        
        print('splits_idx: ', self.splits_idx)
        print('use_10_fold: ', use_10_fold)
        
        # Splits:
        if self.splits_idx is None:
            if 'split_file' in self.config:
                prefix = self.config['split_file']
                splits_filename = self.processed_dir / f"{prefix}_{self.name}_splits.json"
            else:
                splits_filename = self.processed_dir / f"{self.name}_splits.json"
                
            if not splits_filename.exists():
                print('split file not exists, make splits:', splits_filename)
                self.splits = []
                self._make_splits(splits_filename)
                print('make splits, len: ', len(self.splits))
            else:
                self.splits = json.load(open(splits_filename, "r"))
                print('load splits:', splits_filename)
            
            print('split counts:', len(self.splits))
            
        # # TODO: if add more node features:
        if self.additional_features is not None:
            # TODO: pass node function?
            # node register
            self._add_features()

        if self.additional_graph_features is not None:
            self._add_graph_features()

        self.sampler = None
        if 'sampler' in self.config:
            self.sampler = self.config['sampler']
            
            
        # check:
        if is_pyg_dataset(self.name):
            print('pyg data check edge_index: ', self.dataset[0].edge_index)
        else:
            print('check edge_index: ', self.dataset.data[0].edge_index)
            
        # TODO: replace all graphs with random graphs.
        # TODO: ratio:
        if 'perturb_op' in self.config:
            self.replace_edge_index = False if self.config['perturb_op'] is None else True
        else:
            self.replace_edge_index = False
            
        self.edge_index_rewired = None
        self.edge_index_original = None
        
        if self.replace_edge_index:
            self.edge_index_rewired  = []
            self.edge_index_original  = []
            print_log = False
            perturb_op = self.config['perturb_op']
            
            ratios = self.config['rewire_ratio'].split(',')
            ratios = [float(ra) for ra in ratios]
            
            print('Start rewiring edges in ratio: ', ratios)
            
            for i in range(len(self.dataset)):
                # repalce edge_index with random edge_index
                if is_pyg_dataset(self.name):
                    # save original edge_index
                    self.edge_index_original.append(self.dataset[i].edge_index.clone())
                    if self.dataset[i].num_nodes < 4:
                        self.edge_index_rewired.append(self.dataset[i].edge_index.clone())
                        continue
                    # randomly generate 3 edge_index:
                    rewired_edges = [perturb_edge_index(self.dataset[i].num_nodes, self.dataset.__data_list__[i].edge_index, ratio, op=perturb_op) for ratio in ratios]
                    rewired_ones = rewired_edges[0] if len(rewired_edges) == 1 else tuple(rewired_edges)
                    self.edge_index_rewired.append(rewired_ones)
                else:
                    # save original edge_index
                    self.edge_index_original.append(deepcopy(self.dataset.data[i].edge_index))
                    
                    if self.dataset.data[i].N < 4:
                        self.edge_index_rewired.append(self.dataset.data[i].edge_index.clone())
                        continue
                    rewired_edges = [perturb_edge_index(self.dataset[i].num_nodes, self.dataset.data[i].edge_index, ratio, perturb_op) for ratio in ratios]
                    if i == 0:
                        print('rewired_edge 0: ', rewired_edges[0])
                        print('origin edge: ', self.dataset.data[i].edge_index)
                        
                    rewired_ones = rewired_edges[0] if len(rewired_edges) == 1 else tuple(rewired_edges)
                    
                    self.edge_index_rewired.append(rewired_ones)
                    
                    
                if not print_log:
                    print('replace edge_index with random edge_index')
                    print_log = True
                    
            # check:
            print('aft edge_index: ', self.edge_index_rewired[0])
                
    def get_labels(self):
        if is_pyg_dataset(self.name):
            return self.dataset.data.y
        else:
            return [i.y for i in self.dataset]
    
    
    def load_pyg_dataset(self):
        print('self.name:', self.name)
        print(self.name == 'ogbg-ppa')
        if self.name == 'ogbg-ppa':
            edge_attr = False
            if 'edge_attr' in self.config:
                edge_attr = self.config['edge_attr']
            if edge_attr:
                dataset = PygGraphPropPredDataset(name=self.name, root='DATA', transform=add_edge_attr)
            else:
                dataset = PygGraphPropPredDataset(name=self.name, root='DATA', transform=add_zeros)
        else:
            dataset = PygGraphPropPredDataset(name=self.name, root='DATA')
        
        self.splits_idx = dataset.get_idx_split()
        
        if dataset.num_tasks == 1:
            self._target_dim = dataset.num_classes
        else:
            self._target_dim = dataset.num_tasks
            
        print('ogbg _target_dim:', self._target_dim)
        # NOTE: fill __data_list__
        [_ for _ in dataset]
        
        self.targets = dataset.data.y.squeeze().numpy()
        
        return dataset
    
    
    def load_syndataset(self):
        corr = self.corr
        if 'dataset_para' in self.config:
            corr = self.config['dataset_para']
        
        dataset = SynDataset(name=f'{self.name}_{corr}', root='DATA')
        # change name:
        self.name = f'{self.name}_{corr}'
        
        self._target_dim = dataset.num_tasks
        # NOTE: fill __data_list__
        [_ for _ in dataset]
        self.targets = dataset.get_targets()
        return dataset
    
    def load_dataset(self):
        print('load dataset !')
        self.splits_idx = None
        if self.name.startswith('ogbg'):
            dataset = self.load_pyg_dataset()
            
        elif self.name.startswith('syn'):
            dataset = self.load_syndataset()
        else:
            dataset = GraphDataset(torch.load(self.processed_dir / f"{self.name}.pt"))
            
            # NOTE: update _target_dim
            targets = dataset.get_targets()
            
            dims = targets.dim() if isinstance(targets, torch.Tensor) else targets.ndim
            if dims > 1:
                self._target_dim = targets.shape[-1]
            else:
                self._target_dim = np.unique(targets).size
            self.targets = targets
            
        return dataset
    
    @property
    def init_method(self):
        if self.use_random_normal:
            return "random_nomral"

    @property
    def num_graphs(self):
        return len(self.dataset)

    @property
    def target_dim(self):
        if not hasattr(self, "_target_dim") or self._target_dim is None:
            # not very efficient, but it works
            # todo not general enough, we may just remove it
            self._target_dim = np.unique(self.dataset.get_targets()).size
        return self._target_dim

    @property
    def edge_attr_dim(self):
        self._edge_attr_dim = None
        if is_pyg_dataset(self.name):
            if hasattr(self.dataset[0], 'edge_attr') and self.dataset[0].edge_attr is not None:
                if len(self.dataset[0].edge_attr.size()) == 1:
                    self._edge_attr_dim = 1
                else:
                    self._edge_attr_dim = self.dataset[0].edge_attr.shape[1]
        else:
            if hasattr(self.dataset.data[0], 'edge_attr') and self.dataset.data[0].edge_attr is not None:
                if len(self.dataset.data[0].edge_attr.shape) == 1:
                    self._edge_attr_dim = 1
                else:
                    self._edge_attr_dim = self.dataset.data[0].edge_attr.shape[1]
                    
        return self._edge_attr_dim
    
    @property
    def fea_dim(self):
        # TODO: check the graph level features:
        if self.additional_graph_features is not None:
            if is_pyg_dataset(self.name):
                self._fea_dim = self.dataset[0].g_x.shape[-1]
            else:
                self._fea_dim = self.dataset.data[0].g_x.shape[-1]
        else:
            if is_pyg_dataset(self.name):
                self._fea_dim = self.dataset[0].x.shape[-1]
                # self._fea_dim = self.dataset.__data_list__[0].x.shape[-1]
            else:
                self._fea_dim = self.dataset.data[0].x.size(1)
                
        print('input feature dimension: ', self._fea_dim)

        # best for feature initialization based on the current implementation
        # if not hasattr(self, "_fea_dim") or self._fea_dim is None:
        # not very elegant, but it works
        # todo not general enough, we may just remove it
        # self._fea_dim = self.dataset.data[0].x.size(1)
        # feature initialization
        return self._fea_dim

    def get_dense_adjs(self, dataset):
        adjs = []
        if is_pyg_dataset(self.name):
            for d in dataset:
                if d.edge_index.numel() < 1:
                    N = d.x.shape[0]
                    adj = np.ones(shape=(N, N))
                else:
                    adj = torch_utils.to_dense_adj(d.edge_index).numpy()[0]
                adjs.append(adj)
        else:
            adjs = [d.to_numpy_array() for d in dataset.data]
            
        return adjs

    def _add_graph_features(self):
        # TODO: add graph-wise features:
        self.additional_graph_features = self.additional_graph_features.strip().split(',')
        graph_fea_reg = node_feature_utils.GraphFeaRegister()
        for feature_arg in self.additional_graph_features:
            graph_fea_reg.register_by_str(feature_arg)
        self.graph_fea_reg = graph_fea_reg

        adjs = self.get_dense_adjs(self.dataset)
        

        # TODO: load from file if exist, if not exist, then save if it's the first fold test.
        # TODO: save each feature type as separately, e.g., cycle4.pkl, degree.pkl, etc.
        feature_names = self.graph_fea_reg.get_registered()
        graph_features = []
        for ts in feature_names:
            # NOTE: check existence.
            name = ts[0]
            add_features_path = os.path.join(
                self.processed_dir, f'graphwise_{self.name}_add_{name}.pkl')
            if os.path.exists(add_features_path):
                print('load exist graph feature path:', add_features_path)
                with open(add_features_path, 'rb') as f:
                    graph_feature = pk.load(f)
                    print('laod graph_features len: ', len(graph_feature))
                    print('load graph_feature name: ', name)
                    graph_features.append(graph_feature)
                # remove from register_node_features.
                self.graph_fea_reg.remove(name)

        # NOTE: generate rest features:
        if len(self.graph_fea_reg.get_registered()) > 0:
            print('Generate rest features!',
                  self.graph_fea_reg.get_registered())
            rest_graph_features = node_feature_utils.register_features(
                adjs, self.graph_fea_reg)
            # save each
            for i, ts in enumerate(self.graph_fea_reg.get_registered()):
                add_features_path = os.path.join(
                    self.processed_dir, f'graphwise_{self.name}_add_{ts[0]}.pkl')
                graph_features.append(rest_graph_features[i])
                print('rest graph features: ', rest_graph_features[i][0].shape)
                with open(add_features_path, 'wb') as f:
                    pk.dump(rest_graph_features[i], f)
                    print('dump graph features: ', ts[0], 'filepath: ', add_features_path)

        print('_add_graph_features aft:', len(graph_features), ' shape: ',
              graph_features[0][0].shape, graph_features[0][3].shape)

        graph_features = node_feature_utils.composite_graph_feature_list(
            graph_features)
        # 2022.10.20, NOTE: normalize:

        if 'norm_feature' in self.config:
            if self.config['norm_feature']:
                print('Need to normalize graph features !!!!!!!!!!!')
                graph_features = my_utils.normalize(graph_features, along_axis=-1)

        # store in graph as graph not x, but g_x.
        # G6250
        
        if is_pyg_dataset(self.name):
            for i in range(len(self.dataset.__data_list__)):
                d = self.dataset.__data_list__[i]
                d.g_x = torch.FloatTensor(graph_features[i])
                self.dataset.__data_list__[i] = d
        else:
            for i, d in enumerate(self.dataset.data):
                d.set_additional_attr('g_x', torch.FloatTensor(graph_features[i]))

        print('add graph feature done!')


    def _add_additional_features(self, additional_features_list:list):
        if len(additional_features_list) < 1:
            return None
        
        node_fea_reg = node_feature_utils.NodeFeaRegister()
        for feature_arg in additional_features_list:
            if 'degree' in feature_arg:
                feature_arg += f'@name:{self.name}'
            node_fea_reg.register_by_str(feature_arg)
        self.node_fea_reg = node_fea_reg

        # TODO: check padding:
        need_pad = False
        if self.node_fea_reg.contains('kadj'):
            need_pad = True

        # get maximum node num:
        adjs = []
        max_N = 0
        if is_pyg_dataset(self.name):
             for d in self.dataset:
                N = d.num_nodes
                if d.edge_index.numel() < 1:
                    adj = np.ones(shape=(N, N))
                else:
                    # NOTE: if has no edges then failed !!!!
                    adj = torch_utils.to_dense_adj(d.edge_index, max_num_nodes=N).numpy()[0]
                    
                adjs.append(adj)
                max_N = N if N > max_N else max_N
        else:
            for d in self.dataset:
                adjs.append(d.to_numpy_array())
                if max_N < d.N:
                    max_N = d.N

        feature_names = self.node_fea_reg.get_registered()
        node_features = []
        
        additional_feature_path = lambda fea_name,fea_args: \
            os.path.join(self.processed_dir, f'{self.name}_add_{fea_name}_{fea_args}.pkl')
            
        # for ts in feature_names:
        #     name = ts[0]
        #     add_features_path = additional_feature_path(name, ts[-1])
        #     print('additional_features_path: ', add_features_path)
        #     if os.path.exists(add_features_path):
        #         with open(add_features_path, 'rb') as f:
        #             node_feature = pk.load(f)
        #             print('laod node_features len: ', len(node_feature))
        #             print('load node_feature: ', name)
        #             node_features.append(node_feature)
        #         # remove from register_node_features.
        #         self.node_fea_reg.remove(name)

        # NOTE: generate rest node features:
        if len(self.node_fea_reg.get_registered()) > 0:
            print('has rest features:')
            rest_node_features = node_feature_utils.register_features(
                adjs, self.node_fea_reg)
            # TODO: save each
            for i, ts in enumerate(self.node_fea_reg.get_registered()):
                add_features_path = os.path.join(self.processed_dir, f'{self.name}_add_{ts[0]}_{ts[-1]}.pkl')
                node_features.append(rest_node_features[i])
                with open(add_features_path, 'wb') as f:
                    pk.dump(rest_node_features[i], f)
                    # dump node_feature:  DATA/syn_degree/processed/syn_degree_add_degree_{'name': 'syn_degree'}.pkl
                    print('dump node_feature: ', add_features_path)

        print('aft:', len(node_features),
            ' shape: ', node_features[0][0].shape)


        # NOTE: padding
        if need_pad:
            node_features = node_feature_utils.composite_node_feature_list(
                node_features, padding=True, padding_len=max_N+10)
        else:
            node_features = node_feature_utils.composite_node_feature_list(
                node_features, padding=False)

        # 2022.10.20, NOTE: normalize:
        # TODO: normalize through each graph ????
        if 'norm_feature' in self.config:
            if self.config['norm_feature']:
                print('normalize node features!!')
                node_features = my_utils.normalize(
                    node_features, along_axis=-1, same_data_shape=False)
        
        return node_features
    
    def _add_features(self):
        # TODO: load from files, if no files, create ???

        print('adding additional features --')
        all_features = self.additional_features.strip().split(',')
        additional_features_list ,use_features_list = [], []
        for s in all_features:
            if s.startswith('use_'): 
                use_features_list.append(s)
            else: 
                additional_features_list.append(s)
        
        addi_node_features = self._add_additional_features(additional_features_list)

        node_attribute = False
        if 'node_attribute' in self.config:
            node_attribute = self.config['node_attribute']
        print('------- whether use original node_attribute: ', node_attribute)
        
        used_features = None
        if len(use_features_list) > 0:
            used_features = self._save_load_use_features()
            print('used_features len:', len(used_features))
            
        # NOTE: composite [attr, additional, used] those 3 features:
        
        if is_pyg_dataset(self.name):
            data_list = self.dataset.__data_list__
        else:
            data_list = self.dataset.data
        
        for i, d in enumerate(data_list):
            # concatenate with pre features.
            new_x = []
            if node_attribute:
                new_x.append(d.x)
                
            if addi_node_features is not None:
                addi_features = torch.FloatTensor(addi_node_features[i])
                new_x.append(addi_features)
                
            if used_features is not None:
                all_feas = []
                for each_fea in used_features:
                    if len(each_fea) > 0:
                        all_feas.append(torch.FloatTensor(each_fea[i]))
                all_feas = torch.cat(all_feas, dim=-1)
                new_x.append(all_feas)
                
            if len(new_x) == 0:
                raise EmptyNodeFeatureException
                
            data_list[i].x = torch.cat(new_x, axis=-1)
        
        # print('self.dataset.__data_list__: ', self.dataset.__data_list__[0].x.shape)
        # print('self.dataset.data shape: ', self.dataset.data.x.shape)
        # print('next:', next(iter(self.dataset)).x.shape)
            
        # TODO: shuffle x among all node samples.
        if 'shuffle_feature' in self.config:
            if self.config['shuffle_feature']:
                node_num_total = 0
                node_index = {}
                start_id = 0
                copy_degree_sequence = []
                for i, d in enumerate(self.dataset.data):
                    node_num = d.x.shape[0]
                    node_num_total += node_num
                    for j in range(node_num):
                        node_index[start_id] = (i, j)
                        start_id += 1
                        copy_degree_sequence.append(d.x[j].item())
                
                shuf_idx = list(np.arange(node_num_total))
                # np.random.shuffle(shuf_idx)
                pre_value = torch.FloatTensor(self.dataset.data[0].x)
                print('pre_value: ', pre_value)
                
                sample_ids = [s for s in np.random.choice(shuf_idx, size=len(shuf_idx), replace=True)].__iter__()
                # sample_ids = np.random.randint(1, int(4), size=len(shuf_idx)).__iter__()
                
                for d in self.dataset.data:
                    new_x = []
                    N = d.x.shape[0]
                    for i in range(N):
                        new_x.append(copy_degree_sequence[sample_ids.__next__().item()])
                        
                    d.x = torch.FloatTensor(new_x).reshape(N, 1)
                    d.x = torch.FloatTensor(new_x).reshape(N, 1)
                    
                print('replaced value:', self.dataset.data[0].x)
                print(self.dataset.data[0].x - pre_value)
                
                # # construct pairs
                # pairs = []
                # for i in range(0, len(shuf_idx), 2):
                #     if i + 1 < len(shuf_idx):
                #         pairs.append((shuf_idx[i], shuf_idx[i+1]))
                
                # print(f'shuffle feature!, total len: {node_num_total}, pair len: {len(pairs)}')
                # # reconstruct:
                # for (p1, p2) in pairs:
                #     # swich p1 p2 in place
                #     p1_node, p1_x_id = node_index[p1]
                #     p2_node, p2_x_id = node_index[p2]
                #     tmp = self.dataset.data[p1_node].x[p1_x_id]
                #     self.dataset.data[p1_node].x[p1_x_id] = self.dataset.data[p2_node].x[p2_x_id]
                #     self.dataset.data[p2_node].x[p2_x_id] = tmp
                # TODO: how to check correctness???????
                # TODO: dump to file:
                
                
        print('added feature done!')

    def _save_load_use_features(self, graphs=None):
        raise NotImplementedError
    
    def _process(self):
        raise NotImplementedError

    def _download(self):
        raise NotImplementedError

    def _make_splits(self, filename):
        """
        DISCLAIMER: train_test_split returns a SUBSET of the input indexes,
            whereas StratifiedKFold.split returns the indexes of the k subsets, starting from 0 to ...!
        """
        if self.targets is None:
            targets = self.dataset.get_targets()
        else:
            targets = self.targets
            
        all_idxs = np.arange(len(targets))

        if self.outer_k is None:  # holdout assessment strategy
            assert self.holdout_test_size is not None

            if self.holdout_test_size == 0:
                train_o_split, test_split = all_idxs, []
            else:
                outer_split = train_test_split(all_idxs,
                                               stratify=targets,
                                               test_size=self.holdout_test_size)
                train_o_split, test_split = outer_split
            split = {"test": all_idxs[test_split], 'model_selection': []}

            train_o_targets = targets[train_o_split]

            if self.inner_k is None:  # holdout model selection strategy
                if self.holdout_test_size == 0:
                    train_i_split, val_i_split = train_o_split, []
                else:
                    train_i_split, val_i_split = train_test_split(train_o_split,
                                                                  stratify=train_o_targets,
                                                                  test_size=self.holdout_test_size)
                split['model_selection'].append(
                    {"train": train_i_split, "validation": val_i_split})

            else:  # cross validation model selection strategy
                inner_kfold = self.kfold_class(
                    n_splits=self.inner_k, shuffle=True)
                for train_ik_split, val_ik_split in inner_kfold.split(train_o_split, train_o_targets):
                    split['model_selection'].append(
                        {"train": train_o_split[train_ik_split], "validation": train_o_split[val_ik_split]})

            self.splits.append(split)

        else:  # cross validation assessment strategy
            outer_kfold = self.kfold_class(
                n_splits=self.outer_k, shuffle=True)

            print(len(all_idxs))
            print(np.isnan(targets).any())
            print('targets: ', targets.shape)
            
            for train_ok_split, test_ok_split in outer_kfold.split(X=all_idxs, y=targets):
                split = {
                    "test": all_idxs[test_ok_split], 'model_selection': []}

                train_ok_targets = targets[train_ok_split]

                if self.inner_k is None:  # holdout model selection strategy
                    assert self.holdout_test_size is not None
                    train_i_split, val_i_split = train_test_split(train_ok_split,
                                                                  stratify=train_ok_targets,
                                                                  test_size=self.holdout_test_size)
                    split['model_selection'].append(
                        {"train": train_i_split, "validation": val_i_split})

                else:  # cross validation model selection strategy
                    inner_kfold = self.kfold_class(
                        n_splits=self.inner_k, shuffle=True)
                    for train_ik_split, val_ik_split in inner_kfold.split(train_ok_split, train_ok_targets):
                        split['model_selection'].append(
                            {"train": train_ok_split[train_ik_split], "validation": train_ok_split[val_ik_split]})

                self.splits.append(split)

        with open(filename, "w") as f:
            json.dump(self.splits[:], f, cls=NumpyEncoder)

    def _get_loader(self, dataset, batch_size=1, shuffle=True):

        # dataset = GraphDataset(data)
        if self.sampler is None:
            sampler = RandomSampler(dataset) if shuffle is True else None
        elif self.sampler == 'imbalanced':
            sampler = ImbalancedDatasetSampler(dataset)
        else:
            raise NotImplementedError

        # 'shuffle' needs to be set to False when instantiating the DataLoader,
        # because pytorch  does not allow to use a custom sampler with shuffle=True.
        # Since our shuffler is a random shuffler, either one wants to do shuffling
        # (in which case he should instantiate the sampler and set shuffle=False in the
        # DataLoader) or he does not (in which case he should set sampler=None
        # and shuffle=False when instantiating the DataLoader)
        sampler = None
        if is_pyg_dataset(self.name):
            return torch_DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        
        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          shuffle=True,  # if shuffle is not None, must stay false, ow is shuffle is false
                          pin_memory=True)
    
    
    def to_orignal_edge_index(self, dataset, idxs):
        if len(dataset) == len(idxs):
            # pyg dataset
            for i in range(len(idxs)):
                dataset[i].edge_index = self.edge_index_original[idxs[i]].clone()
        else:
            for i in range(len(idxs)):
                dataset[idxs[i]].edge_index = self.edge_index_original[idxs[i]].clone()
        return dataset
    
    
    def to_rewired_edge_index(self, dataset, idxs):
        if len(dataset) == len(idxs):
            for i in range(len(idxs)):
                to_rewired = self.edge_index_rewired[idxs[i]]
                dataset[i].edge_index = to_rewired
        else:
            for i in range(len(idxs)):
                to_rewired = self.edge_index_rewired[idxs[i]]
                dataset[idxs[i]].edge_index = to_rewired
        return dataset

    def get_test_fold(self, outer_idx, batch_size=1, shuffle=True):
        
        # NOTE: if ogb then return:
        outer_idx = outer_idx or 0
        
        if is_pyg_dataset(self.name):
            if self.splits_idx is not None:
                split_idx = self.splits_idx['test']
            else:
                split_idx = self.splits[outer_idx]["test"]
            
            test_dataset = self.dataset[split_idx]
            if self.edge_index_rewired:
                # replace the edge_index with the rewired one
                print('using rewired edge to test')
                test_dataset = self.to_rewired_edge_index(test_dataset, split_idx)
                
            test_loader = self._get_loader(test_dataset, batch_size, shuffle=False)
            return test_loader
        
        if outer_idx - 1 > len(self.splits):
            return None, None

        idxs = self.splits[outer_idx]["test"]
        
        test_data = self.dataset.get_data()
        # check edge_index:
        print('before test edge index: ', test_data[idxs[0]].edge_index)
        
        if self.edge_index_rewired is not None:
            test_data = self.to_rewired_edge_index(test_data, idxs)
            print('after using rewired edge to test edge index: ', test_data[idxs[0]].edge_index)
            
        test_dataset = GraphDatasetSubset(test_data, idxs)
        
        if len(test_dataset) == 0:
            test_loader = None
        else:
            test_loader = self._get_loader(test_dataset, batch_size, shuffle=False)

        return test_loader

    
    def get_model_selection_fold(self, outer_idx, inner_idx=None, batch_size=1, shuffle=True):
        # NOTE: if ogb then return:
                
        outer_idx = outer_idx or 0
        inner_idx = inner_idx or 0
        
        if is_pyg_dataset(self.name):
            if self.splits_idx is not None:
                train_dataset = self.dataset[self.splits_idx['train']]
                val_dataset = self.dataset[self.splits_idx['valid']]
                
                if self.replace_edge_index:
                    train_dataset = self.to_rewired_edge_index(train_dataset, self.splits_idx['train'])
                    val_dataset = self.to_rewired_edge_index(val_dataset, self.splits_idx['valid'])
                    
                train_loader = self._get_loader(train_dataset, batch_size, shuffle)
                val_loader = self._get_loader(val_dataset, batch_size, False)
                
                return train_loader, val_loader
            else:
                idxs = self.splits[outer_idx]["model_selection"][inner_idx]
                
                print(type(idxs))
                print(len(idxs))
                print(type(idxs["train"]))
                print(len(idxs["train"]))
                print(idxs["train"][0])
                
                train_dataset = self.dataset[idxs["train"]]
                val_dataset = self.dataset[idxs['validation']]

                if self.replace_edge_index:
                    train_dataset = self.to_rewired_edge_index(train_dataset, self.splits_idx['train'])
                    val_dataset = self.to_rewired_edge_index(val_dataset, self.splits_idx['valid'])

                
                train_loader = self._get_loader(train_dataset, batch_size, shuffle)
                val_loader = self._get_loader(val_dataset, batch_size, False)
                
                return train_loader, val_loader
                
        if outer_idx - 1 > len(self.splits):
            return None, None
        
        idxs = self.splits[outer_idx]["model_selection"][inner_idx]
        
        all_data = self.dataset.get_data()

        if self.replace_edge_index:
            all_data = self.to_rewired_edge_index(all_data, idxs["train"])
            all_data = self.to_rewired_edge_index(all_data, idxs["validation"])
        
        train_data = GraphDatasetSubset(all_data, idxs["train"])
        val_data = GraphDatasetSubset(all_data, idxs["validation"])

        train_loader = self._get_loader(train_data, batch_size, shuffle)

        if len(val_data) == 0:
            val_loader = None
        else:
            val_loader = self._get_loader(val_data, batch_size, shuffle)
        
        return train_loader, val_loader


class OGBMoleculeDatasetManager(GraphDatasetManager):
    classfication =True

    def _download(self):
        # TODO: write to file
        """
        name (string): The name of the dataset (one of :obj:`"PATTERN"`,
            :obj:`"CLUSTER"`, :obj:`"MNIST"`, :obj:`"CIFAR10"`,
            :obj:`"TSP"`, :obj:`"CSL"`)
            
            ogbg-molbace
            ogbg-molbbbp
            ogbg-molclintox
            ogbg-molmuv
            ogbg-molpcba
            ogbg-molsider
            ogbg-moltox21
            ogbg-moltoxcast
            ogbg-molhiv
            ogbg-molesol
            ogbg-molfreesolv
            ogbg-mollipo
            ogbg-molchembl
            ogbg-ppa
            ogbg-code2
            
        Beside the two main datasets, we additionally provide 10 smaller datasets from MoleculeNet. 
        They are ogbg-moltox21, ogbg-molbace, ogbg-molbbbp, ogbg-molclintox, ogbg-molmuv, ogbg-molsider, 
        and ogbg-moltoxcast for (multi-task) binary classification, and ogbg-molesol, ogbg-molfreesolv, and ogbg-mollipo 
        for regression. Evaluators are also provided for these datasets. These datasets can be used to stress-test molecule-specific 
        methods or transfer learning [4].
        """
        # elif args.dataset == 'ogbg-ppa':
        # if args.dataset == 'ogbg-molhiv':
        print('self. name:', self.name)
        dataset = PygGraphPropPredDataset(name=self.name, root='dataset')
        # dataset = PygGraphPropPredDataset(name=args.dataset, root=ar, transform=add_zeros)
        del dataset
        print('Downloaded')

    def _process(self):
        # TODO: combine trian, val, test:
        # load from raw:
        print('do nothing for pyg dataset')
        # dataset = PygGraphPropPredDataset(name=self.name, root='DATA')
        # print(f'len: {len(dataset)}')
        # all_data = [Data.from_pyg_data(d) for d in dataset]

        # torch.save(all_data, self.processed_dir / f"{self.name}.pt")
        # print(f"saved: {self.processed_dir} / saved : {self.name}.pt")
        # split_idx = dataset.get_idx_split()

        # if 'mol_split' in self.config:
        #     if self.config['mol_split']:
        #         tr = [i.item() for i in split_idx['train'].numpy()]
        #         vl= [i.item() for i in split_idx['valid'].numpy()]
        #         te = [i.item() for i in split_idx['test'].numpy()]

        #         splits = [{ "test": te,
        #                     'model_selection': [{'train':tr,
        #                                         "validation":vl}]}]
                
        #         filename = self.processed_dir / f"{self.name}_splits.json"
        #         with open(filename, "w") as f:
        #             json.dump(splits, f, cls=NumpyEncoder)

        #         print(f"mol splits saved: {filename}")
        # else:
        #     print('not use mol split')

    
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from rdkit.Chem.rdmolops import FastFindRings

class MoleculeDatasetManager(GraphDatasetManager):
    classification = True

    def _scaffold_split(self, dataset, scaffold_func='decompose'):
     

        smiles = [d.smiles for d in dataset]
        molecules = [Chem.MolFromSmiles(s, sanitize=None) for s in smiles]

        def get_scaffold_sets(molecules, scaffold_func):
            res_dict = {}
            if scaffold_func == 'decompose':
                for i, mol in enumerate(molecules):
                    FastFindRings(mol)
                    res_dict[i] = Chem.MolToSmiles(AllChem.MurckoDecompose(mol))
            elif scaffold_func == 'smiles':
                for i, mol in enumerate(molecules):
                    FastFindRings(mol)
                    res_dict[i] =MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) 
            else:
                raise NotImplementedError
            return res_dict

        scaffold_dict = get_scaffold_sets(molecules, scaffold_func)
        scaffolds_set = list(set(scaffold_dict.values()))
   
        train_scaffolds, test_scaffolds = train_test_split(scaffolds_set, test_size=0.1)
        train_scaffolds, val_scaffolds = train_test_split(train_scaffolds, test_size=0.1)

        train_indices = [i for i in range(len(molecules)) if scaffold_dict[i] in train_scaffolds]
        test_indices = [i for i in range(len(molecules))  if scaffold_dict[i] in test_scaffolds]
        val_indices = [i for i in range(len(molecules)) if scaffold_dict[i] in val_scaffolds]
        # scaffold split, train : 32462, test: 3702, val:4963, total:41127
        print(f'scaffold split, train : {len(train_indices)}, test: {len(test_indices)}, val:{len(val_indices)}')
        return train_indices, test_indices, val_indices
    

    # def _random_folds(self, dataset):
    #     # NOTE:split along y
    #     label_dict = defaultdict(list)
    #     for i, d in enumerate(dataset):
    #         label_dict[d.y.item()].append(i)

    #     train_splits, test_splits, val_splits = [], [], []
    #     for _, v in label_dict.items():
    #         tr, te = train_test_split(v, test_size=0.1)
    #         train_splits.extend(tr)
    #         test_splits.extend(te)
    #         tr, vl = train_test_split(tr, test_size=0.1)
    #         train_splits.extend(tr)
    #         val_splits.extend(vl)
        
    #     return train_splits, test_splits, val_splits



    def _download(self):
        # TODO: write to file
        """
        name (string): The name of the dataset (one of :obj:`"PATTERN"`,
            :obj:`"CLUSTER"`, :obj:`"MNIST"`, :obj:`"CIFAR10"`,
            :obj:`"TSP"`, :obj:`"CSL"`)
        """
        MoleculeNet(root='DATA', name=self.name)
        print('Downloaded')

    def _process(self):
        # TODO: combine trian, val, test:
        # load from raw:
        cur_dataset = MoleculeNet(root='DATA', name=self.name)
        print(f'len: {len(cur_dataset)}')
        all_data = [Data.from_pyg_data(d) for d in cur_dataset]

        torch.save(all_data, self.processed_dir / f"{self.name}.pt")
        print(f"saved: {self.processed_dir} / saved : {self.name}.pt")

        if 'mol_split' in self.config:
            if self.config['mol_split']:
                splits = []
                for _ in range(self.outer_k):
                    tr, te, vl = self._scaffold_split(cur_dataset)
                    splits.append({ "test": te, 
                            'model_selection': [{'train':tr,
                                                "validation":vl}]})
                

                filename = self.processed_dir / f"{self.name}_splits.json"
                with open(filename, "w") as f:
                    json.dump(splits, f, cls=NumpyEncoder)

                print(f"mol splits saved: {filename}")


class GNNBenchmarkDatasetManager(GraphDatasetManager):
    classification = True

    def _download(self):
        # TODO: write to file
        """
        name (string): The name of the dataset (one of :obj:`"PATTERN"`,
            :obj:`"CLUSTER"`, :obj:`"MNIST"`, :obj:`"CIFAR10"`,
            :obj:`"TSP"`, :obj:`"CSL"`)
        """
        split_name = ['train', 'val', 'test']
        for n in split_name:
            GNNBenchmarkDataset(root='DATA', name=self.name, split=n)
            print('Downloaded')

    def _process(self):
        # TODO: combine trian, val, test:
        # load from raw:
        split_name = ['train', 'val', 'test']
        splits = []
        all_data = []
        for n in split_name:
            cur_dataset = GNNBenchmarkDataset(root='DATA', name=self.name, split=n)
            print(f'{n} len: {len(cur_dataset)}')
            splits.append(len(cur_dataset))
            all_data.extend([MyData.from_pyg_data(d) for d in cur_dataset])
        
        torch.save(all_data, self.processed_dir / f"{self.name}.pt")
        print(f"saved: {self.processed_dir} / saved : {self.name}.pt")

        # TODO: ignore splits:
        
        # split = [{ "test": [i for i in range(splits[0]+splits[1], len(all_data))], 
        #          'model_selection': [{'train':[i for i in range(0, splits[0])],
        #                               "validation":[i for i in range(splits[0], splits[0]+splits[1])]}]}]
        
        # filename = self.processed_dir / f"{self.name}_splits.json"
        # with open(filename, "w") as f:
        #     json.dump(split, f, cls=NumpyEncoder)

        # print(f"saved: {filename}")

from pathlib import Path



class PlanetoidDatasetManager(GraphDatasetManager):
    classification = True

    def __init__(self, name):
        self.name = name
        self.processed_dir = Path('DATA/processed')
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _download(self):
        # Download the dataset
        Planetoid(root='DATA', name=self.name)
        print('Downloaded')

    def _process(self):
        # Load the dataset
        dataset = Planetoid(root='DATA', name=self.name)
        all_data = [Data.from_pyg_data(d) for d in dataset]
        
        torch.save(all_data, self.processed_dir / f"{self.name}.pt")
        print(f"saved: {self.processed_dir} / saved : {self.name}.pt")

    
    
    
class PPIDatasetManager(GraphDatasetManager):
    classification = True

    def _download(self):
        # TODO: write to file
        split_name = ['train', 'val', 'test']
        for n in split_name:
            PPIDataset(root='DATA', split=n)
            print('Downloaded')

    def _process(self):
        # TODO: combine trian, val, test:
        # load from raw:
        split_name = ['train', 'val', 'test']
        splits = []
        all_data = []
        for n in split_name:
            cur_dataset = PPIDataset(root='DATA', split=n)
            print(f'{n} len: {len(cur_dataset)}')
            splits.append(len(cur_dataset))
            all_data.extend([Data.from_pyg_data(d) for d in cur_dataset])
        
        torch.save(all_data, self.processed_dir / f"{self.name}.pt")
        print(f"saved: {self.processed_dir} / saved : {self.name}.pt")

        split = [{ "test": [i for i in range(splits[0]+splits[1], len(all_data))], 
                 'model_selection': [{'train':[i for i in range(0, splits[0])],
                                      "validation":[i for i in range(splits[0], splits[0]+splits[1])]}]}]
        
        filename = self.processed_dir / f"{self.name}_splits.json"
        with open(filename, "w") as f:
            json.dump(split, f, cls=NumpyEncoder)

        print(f"saved: {filename}")


class TUDatasetManager(GraphDatasetManager):
    URL = "https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
    
    "PTC_MR.zip"
    "QM9.zip"
    
    classification = True

    def _download(self):
        url = self.URL.format(name=self.name)
        response = requests.get(url)
        print(f"Downloading {self.name} dataset from {url}")
        stream = io.BytesIO(response.content)
        with zipfile.ZipFile(stream) as z:
            for fname in z.namelist():
                z.extract(fname, self.raw_dir)

    def _process(self):
        graphs_data, num_node_labels, num_edge_labels, Graph_whole = parse_tu_data(
            self.name, self.raw_dir)  # Graph_whole contains all nodes and edges in the dataset
        targets = graphs_data.pop("graph_labels")

        self.Graph_whole = Graph_whole
        print("in _process")
        # TODO, NOTE: whole graph level !!!
        if self.use_pagerank:
                self.Graph_whole_pagerank = nx.pagerank(self.Graph_whole)
        if self.use_eigen or self.use_eigen_norm:
            try:
                print("{name}".format(name=self.name))
                if self.use_eigen:
                    self.Graph_whole_eigen = np.load(
                        "DATA/{name}_eigenvector.npy".format(name=self.name))
                else:
                    self.Graph_whole_eigen = np.load(
                        "DATA/{name}_eigenvector_degree_normalized.npy".format(name=self.name))
                print('eigen shape:', self.Graph_whole_eigen.shape)
            except:
                num_node = get_dataset_node_num(self.name)
                adj_matrix = nx.to_numpy_array(self.Graph_whole)
                if self.use_eigen_norm:
                    # normalize adjacency matrix with degree
                    sum_of_rows = adj_matrix.sum(axis=1)
                    normalized_adj_matrix = np.zeros((num_node, num_node))
                    # deal with edge case of disconnected node:
                    for i in range(num_node):
                        if sum_of_rows[i] != 0:
                            normalized_adj_matrix[i, :] = adj_matrix[i,
                                                                     :] / sum_of_rows[i, None]
                    adj_matrix = normalized_adj_matrix
                print("start computing eigen vectors")
                w, v = LA.eig(adj_matrix)
                indices = np.argsort(w)[::-1]
                v = v.transpose()[indices]
                # only save top 200 eigenvectors
                if self.use_eigen:
                    np.save(
                        "DATA/{name}_eigenvector".format(name=self.name), v[:200])
                else:
                    np.save(
                        "DATA/{name}_eigenvector_degree_normalized".format(name=self.name), v[:200])
                self.Graph_whole_eigen = v
                print('eigen shape:', self.Graph_whole_eigen.shape)

            print('Graph_whole_eigen: ', self.Graph_whole_eigen)
            print('nonzero: ', np.count_nonzero(self.Graph_whole_eigen == 0))
            
            node_num = get_dataset_node_num(self.name)
            # why top 50????
            
            
            embedding = np.zeros((node_num, 50))
            for i in range(node_num):
                for j in range(50):
                    embedding[i, j] = self.Graph_whole_eigen[j, i]
            self.Graph_whole_eigen = embedding
            print(self.Graph_whole_eigen)
        if self.use_1hot:
            self.onehot = nn.Embedding(self.Graph_whole.number_of_nodes(), 64)

        if self.use_deepwalk:
            self.deepwalk = self.extract_deepwalk_embeddings(
                    "DATA/proteins.embeddings")

        if self.use_random_normal:
            num_of_nodes = self.Graph_whole.number_of_nodes()
            self.rn = np.random.normal(0, 1, (num_of_nodes, 50))

        # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
        max_num_nodes = max([len(v)
                            for (k, v) in graphs_data['graph_nodes'].items()])
        setattr(self, 'max_num_nodes', max_num_nodes)

        dataset = []
        graphs = []
        for i, target in enumerate(targets, 1):
            graph_data = {k: v[i] for (k, v) in graphs_data.items()}
            G = create_graph_from_tu_data(
                graph_data, target, num_node_labels, num_edge_labels, Graph_whole)

            if self.precompute_kron_indices:
                laplacians, v_plus_list = self._precompute_kron_indices(G)
                G.laplacians = laplacians
                G.v_plus = v_plus_list

            if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
                # TODO: convert to numpy : npy
                data = self._to_data(G)
                # TODO save here:
                dataset.append(data)
                G.__class__()
                graphs.append(G)
        
        # Save
        self._save_load_use_features(graphs=graphs)
        for d in dataset:
            print('d type before save:', type(d))        
            print(d.to_numpy_array())
            break
        torch.save(dataset, self.processed_dir / f"{self.name}.pt")
        print(f"saved: {self.processed_dir}/{self.name}.pt")
        
    def _save_load_use_features(self, graphs=None) -> list:
        res = []
        pagerank_fea, onehot_fea, random_fea, eigen_fea, deepwalk_fea = [], [], [], [], []

        def print_fea_info(str1, fea_path, fea:np.ndarray):
            if isinstance(fea, np.ndarray):
                print(f'{str1}: {fea_path}, shape: {fea.shape}')
            else:
                print(f'{str1}: {fea_path}, len: {len(fea)}, shape: {fea[0].shape}')
            
            
        if self.use_pagerank:
            save_dir = f'DATA/{self.name}_tensor_pagerank.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        feas.append([self.Graph_whole_pagerank[node]] * 50)
                    pagerank_fea.append(np.array(feas))
                    
                pk.dump(pagerank_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor:', save_dir, pagerank_fea)
            else:
                pagerank_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor:', save_dir, pagerank_fea)
                
            res.append(pagerank_fea)
         
        if self.use_1hot:
            save_dir = f'DATA/{self.name}_tensor_onehot.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        arr = self.onehot(torch.LongTensor([node-1]))
                        feas.append(list(arr.view(-1).detach().numpy())) 
                    onehot_fea.append(np.array(feas))
                    
                pk.dump(onehot_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor', save_dir, onehot_fea)
            else:
                onehot_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor', save_dir, onehot_fea)
                
            res.append(onehot_fea)
            
        if self.use_random_normal:
            save_dir = f'DATA/{self.name}_tensor_random.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        arr = self.rn[node-1, :]
                        feas.append(list(arr)) # [1,...,50]
                    feas = np.array(feas)
                    random_fea.append(feas) # (N, 50)
                    
                pk.dump(random_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor:', save_dir, random_fea)
            else:
                random_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor:', save_dir, random_fea)
                
                
            res.append(random_fea)
            
        if self.use_eigen:
            save_dir = f'DATA/{self.name}_tensor_eigen.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        feas.append(list(self.Graph_whole_eigen[node-1]))
                    eigen_fea.append(np.array(feas))
                    
                pk.dump(eigen_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor:', save_dir, eigen_fea)
            else:
                eigen_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor:', save_dir, eigen_fea)
                
            res.append(eigen_fea)
            
        if self.use_deepwalk:
            save_dir = f'DATA/{self.name}_tensor_deepwalk.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        feas.append(list(self.deepwalk[node-1]))
                    deepwalk_fea.append(np.array(feas))
                    
                pk.dump(deepwalk_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor:', save_dir, deepwalk_fea)
            else:
                deepwalk_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor:', save_dir, deepwalk_fea)
        
            res.append(deepwalk_fea)
        
        return res

    def _to_data(self, G) -> MyData:
        datadict = {}
        # embedding = None
        # if self.use_1hot:
        #     embedding = self.Graph_whole_embedding
        # elif self.use_random_normal:
        #     embedding = self.Graph_whole_embedding
        # elif self.use_pagerank:
        #     # embedding is essentially pagerank dictionary
        #     embedding = self.Graph_whole_pagerank
        # elif self.use_eigen:
        #     embedding = self.Graph_whole_eigen
        # elif self.use_deepwalk:
        #     embedding = self.Graph_whole_deepwalk
        # TODO: only save attributes

        node_features = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one,
                                self.use_shared, self.use_1hot, self.use_random_normal, self.use_pagerank,
                                self.use_eigen, self.use_deepwalk)
        datadict.update(x=node_features)

        if G.laplacians is not None:
            datadict.update(laplacians=G.laplacians)
            datadict.update(v_plus=G.v_plus)

        edge_index = G.get_edge_index()
        datadict.update(edge_index=edge_index)

        if G.has_edge_attrs:
            edge_attr = G.get_edge_attr()
            datadict.update(edge_attr=edge_attr)

        target = G.get_target(classification=self.classification)
        datadict.update(y=target)

        data = MyData(**datadict)

        return data

    def _precompute_kron_indices(self, G):
        laplacians = []  # laplacian matrices (represented as 1D vectors)
        v_plus_list = []  # reduction matrices

        X = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
        lap = torch.Tensor(normalized_laplacian_matrix(
            G).todense())  # I - D^{-1/2}AD^{-1/2}
        # print(X.shape, lap.shape)

        laplacians.append(lap)

        for _ in range(self.KRON_REDUCTIONS):
            if lap.shape[0] == 1:  # Can't reduce further:
                v_plus, lap = torch.tensor([1]), torch.eye(1)
                # print(lap.shape)
            else:
                v_plus, lap = self._vertex_decimation(lap)
                # print(lap.shape)
                # print(lap)

            laplacians.append(lap.clone())
            v_plus_list.append(v_plus.clone().long())

        return laplacians, v_plus_list

    # For the Perron–Frobenius theorem, if A is > 0 for all ij then the leading eigenvector is > 0
    # A Laplacian matrix is symmetric (=> diagonalizable)
    # and dominant eigenvalue (true in most cases? can we enforce it?)
    # => we have sufficient conditions for power method to converge
    def _power_iteration(self, A, num_simulations=30):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = torch.rand(A.shape[1]).unsqueeze(dim=1) * 0.5 - 1

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            b_k1 = torch.mm(A, b_k)

            # calculate the norm
            b_k1_norm = torch.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k

    def _vertex_decimation(self, L):

        max_eigenvec = self._power_iteration(L)
        v_plus, v_minus = (max_eigenvec >= 0).squeeze(
        ), (max_eigenvec < 0).squeeze()

        # print(v_plus, v_minus)

        # diagonal matrix, swap v_minus with v_plus not to incur in errors (does not change the matrix)
        if torch.sum(v_plus) == 0.:  # The matrix is diagonal, cannot reduce further
            if torch.sum(v_minus) == 0.:
                assert v_minus.shape[0] == L.shape[0], (v_minus.shape, L.shape)
                # I assumed v_minus should have ones, but this is not necessarily the case. So I added this if
                return torch.ones(v_minus.shape), L
            else:
                return v_minus, L

        L_plus_plus = L[v_plus][:, v_plus]
        L_plus_minus = L[v_plus][:, v_minus]
        L_minus_minus = L[v_minus][:, v_minus]
        L_minus_plus = L[v_minus][:, v_plus]

        L_new = L_plus_plus - \
            torch.mm(torch.mm(L_plus_minus, torch.inverse(
                L_minus_minus)), L_minus_plus)

        return v_plus, L_new

    def _precompute_assignments(self):
        pass

    def extract_deepwalk_embeddings(self, filename):
        print("start to load embeddings")
        node_num = get_dataset_node_num(self.name)
        with open(filename) as f:
            feat_data = []
            for i, line in enumerate(f):
                info = line.strip().split()
                if i == 0:
                    feat_data = np.zeros((node_num, int(info[1])))
                else:
                    idx = int(info[0]) - 1
                    feat_data[idx, :] = list(map(float, info[1::]))

        print("finished loading deepwalk embeddings")
        return feat_data



class SynDataset(InMemoryDataset):
    def __init__(self, data=None, name=None, root=None, transform=None, pre_transform=None):
        super(SynDataset, self).__init__(root, transform, pre_transform)
        if data is None:
            data_path = os.path.join(root, f'{name}.pkl')
            print('SynDataset load data_path:', data_path)
            with open(data_path, 'rb') as f:
                data = pk.load(f)
                
        self.num_tasks = len({int(i.y.item()) for i in data})
        self.data, self.slices = self.collate(data)
        self.name = name
        self.root = root
    
    def get_targets(self):
        if self.__data_list__[0].y.shape[0] > 1:
            return np.stack([d.y for d in self.__data_list__], axis=0)
        else:
            return np.array([d.y.item() for d in self.__data_list__])

    def _download(self):
        pass

    def _process(self):
        pass


    
    
class SyntheticManager(TUDatasetManager):
    def _download(self):
        if self.name == 'CSL':
            graphs = generate_CSL(each_class_num=150, N=41, S=[2, 3, 4, 7])
        elif self.name == 'MDG':
            graphs = generate_mix_degree_graphs()
        else:
            raise NotImplementedError

        labels = []
        G = []
        for (g, y) in graphs:
            G.append(g)
            labels.append(y)
        print('dataset name:', self.name, 'len of samples:', len(G))
        # TODO: save G only, test the pickle file size:
        pk.dump(G, open(f'{self.raw_dir}/{self.name}_graph.pkl', 'wb'))
        pk.dump(labels, open(f'{self.raw_dir}/{self.name}_label.pkl', 'wb'))
        print('saved pkl data')

    def _process(self):
        graph_nodes = defaultdict(list)
        graph_edges = defaultdict(list)
        node_labels = defaultdict(list)
        node_attrs = defaultdict(list)
        edge_labels = defaultdict(list)
        edge_attrs = defaultdict(list)

        graphs = pk.load(open(f'{self.raw_dir}/{self.name}_graph.pkl', 'rb'))
        graph_labels = pk.load(
            open(f'{self.raw_dir}/{self.name}_label.pkl', 'rb'))

        for i, g in enumerate(graphs):
            graph_nodes[i] = g.nodes
            graph_edges[i] = g.edges

        graphs_data = {
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "graph_labels": graph_labels,
            "node_labels": node_labels,
            "node_attrs": node_attrs,
            "edge_labels": edge_labels,
            "edge_attrs": edge_attrs
        }

        print("in _process")

        if self.use_pagerank:
            # TODO: create whole graphs:
            Graph_whole = nx.disjoint_union_all(graphs)
            self.Graph_whole = Graph_whole
            self.Graph_whole_pagerank = nx.pagerank(self.Graph_whole)
        elif self.use_eigen or self.use_eigen_norm:
            try:
                print("{name}".format(name=self.name))
                if self.use_eigen:
                    self.Graph_whole_eigen = np.load(
                        "DATA/{name}_eigenvector.npy".format(name=self.name))
                else:
                    self.Graph_whole_eigen = np.load(
                        "DATA/{name}_eigenvector_degree_normalized.npy".format(name=self.name))
                print(self.Graph_whole_eigen.shape)
            except:
                num_node = get_dataset_node_num(self.name)
                adj_matrix = nx.to_numpy_array(self.Graph_whole)
                if self.use_eigen_norm:
                    # normalize adjacency matrix with degree
                    sum_of_rows = adj_matrix.sum(axis=1)
                    normalized_adj_matrix = np.zeros((num_node, num_node))
                    # deal with edge case of disconnected node:
                    for i in range(num_node):
                        if sum_of_rows[i] != 0:
                            normalized_adj_matrix[i, :] = adj_matrix[i,
                                                                     :] / sum_of_rows[i, None]
                    adj_matrix = normalized_adj_matrix
                print("start computing eigen vectors")
                w, v = LA.eig(adj_matrix)
                indices = np.argsort(w)[::-1]
                v = v.transpose()[indices]
                # only save top 200 eigenvectors
                if self.use_eigen:
                    np.save(
                        "DATA/{name}_eigenvector".format(name=self.name), v[:200])
                else:
                    np.save(
                        "DATA/{name}_eigenvector_degree_normalized".format(name=self.name), v[:200])
                self.Graph_whole_eigen = v

            print(self.Graph_whole_eigen)
            print(np.count_nonzero(self.Graph_whole_eigen == 0))
            node_num = get_dataset_node_num(self.name)
            embedding = np.zeros((node_num, 50))
            for i in range(node_num):
                for j in range(50):
                    embedding[i, j] = self.Graph_whole_eigen[j, i]
            self.Graph_whole_eigen = embedding
            print(self.Graph_whole_eigen)
        elif self.use_1hot:
            # TODO: create whole graphs:
            Graph_whole = nx.disjoint_union_all(graphs)
            self.Graph_whole = Graph_whole

            self.Graph_whole_embedding = nn.Embedding(
                self.Graph_whole.number_of_nodes(), 64)
        elif self.use_deepwalk:
            self.Graph_whole_deepwalk = self.extract_deepwalk_embeddings(
                "DATA/proteins.embeddings")
        elif self.use_random_normal:
            # TODO: create whole graphs:
            Graph_whole = nx.disjoint_union_all(graphs)
            self.Graph_whole = Graph_whole
            num_of_nodes = self.Graph_whole.number_of_nodes()
            self.Graph_whole_embedding = np.random.normal(
                0, 1, (num_of_nodes, 50))
        else:
            print('use other base node features! e.g., degree')

        # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
        max_num_nodes = max([len(v)
                            for (k, v) in graphs_data['graph_nodes'].items()])
        setattr(self, 'max_num_nodes', max_num_nodes)

        dataset = []
        for i, g in enumerate(graphs):
            G = create_graph_from_nx(g, graph_labels[i])
            if self.precompute_kron_indices:
                laplacians, v_plus_list = self._precompute_kron_indices(G)
                G.laplacians = laplacians
                G.v_plus = v_plus_list

            if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
                data = self._to_data(G)
                dataset.append(data)
                G.__class__()
        torch.save(dataset, self.processed_dir / f"{self.name}.pt")

        """
        name (string): The name of the dataset (one of :obj:`"PATTERN"`,
            :obj:`"CLUSTER"`, :obj:`"MNIST"`, :obj:`"CIFAR10"`,
            :obj:`"TSP"`, :obj:`"CSL"`)
        """




class MNIST(GNNBenchmarkDatasetManager):
    name = "MNIST"
    _fea_dim = 3
    _target_dim = 10
    

class CIFAR10(GNNBenchmarkDatasetManager):
    name = "CIFAR10"
    _fea_dim = 3
    _target_dim = 10

"""
        # elif args.dataset == 'ogbg-ppa':
        # if args.dataset == 'ogbg-molhiv':
"""


class OGBBBBP(OGBMoleculeDatasetManager):
    name = 'ogbg-molbbbp'
    _fea_dim = 9
    _target_dim = None
    
class OGBBACE(OGBMoleculeDatasetManager):
    name = 'ogbg-molbace'
    _fea_dim = 9
    _target_dim = None

    
    
class OGBTox21(OGBMoleculeDatasetManager):
    name = 'ogbg-moltox21'
    _fea_dim = 9
    _target_dim = None

    
class OGBHIV(OGBMoleculeDatasetManager):
    name = 'ogbg-molhiv'
    _fea_dim = 9
    _target_dim = None


class OGBPPA(OGBMoleculeDatasetManager):
    name = 'ogbg-ppa'
    _fea_dim = 9
    _target_dim = None


class HIV(MoleculeDatasetManager):
    name = "hiv"
    _fea_dim = 9
    _target_dim = 2

class BACE(MoleculeDatasetManager):
    name = "bace"
    _fea_dim = 3
    _target_dim = 2

class BBPB(MoleculeDatasetManager):
    name = "bbpb"
    _fea_dim = 3
    _target_dim = 2


class PPI(PPIDatasetManager):
    name = "PPI"
    _fea_dim = 50
    _target_dim = 121

    "PTC_MR.zip"
    "QM9.zip"
    
class PTC(TUDatasetManager):
    name = "PTC_FM"
    _fea_dim = 37
    _target_dim = 2
    max_num_nodes = 111
    
class QM9(TUDatasetManager):
    name = "QM9"
    _fea_dim = 37
    _target_dim = 2
    max_num_nodes = 111
    
    
class NCI1(TUDatasetManager):
    name = "NCI1"
    _fea_dim = 37
    _target_dim = 2
    max_num_nodes = 111


class AIDS(TUDatasetManager):
    name = "AIDS"
    _fea_dim = 4
    _target_dim = 2
    max_num_nodes = 3782


class RedditBinary(TUDatasetManager):
    name = "REDDIT-BINARY"
    _fea_dim = 1
    _target_dim = 2
    max_num_nodes = 3782


class Reddit5K(TUDatasetManager):
    name = "REDDIT-MULTI-5K"
    _fea_dim = 1
    _target_dim = 5
    max_num_nodes = 3648


class Proteins(TUDatasetManager):
    name = "PROTEINS_full"
    _fea_dim = 3
    _target_dim = 2
    max_num_nodes = 620


class DD(TUDatasetManager):
    name = "DD"
    _fea_dim = 89
    _target_dim = 2
    max_num_nodes = 5748


class Enzymes(TUDatasetManager):
    name = "ENZYMES"
    _fea_dim = 20
    _target_dim = 6
    max_num_nodes = 126


class IMDBBinary(TUDatasetManager):
    name = "IMDB-BINARY"
    _fea_dim = 50
    _target_dim = 2
    max_num_nodes = 136


class IMDBMulti(TUDatasetManager):
    name = "IMDB-MULTI"
    _fea_dim = 1
    _target_dim = 3
    max_num_nodes = 89


class Collab(TUDatasetManager):
    name = "COLLAB"
    _fea_dim = 1
    _target_dim = 3
    max_num_nodes = 492


class Mutag(TUDatasetManager):
    name = "MUTAG"
    _fea_dim = 71
    _target_dim = 2
    max_num_nodes = 100


class CSL(SyntheticManager):
    name = "CSL"
    _fea_dim = 1
    _target_dim = 4
    max_num_nodes = 41


class MDG(SyntheticManager):
    name = "MDG"
    _fea_dim = 1
    _target_dim = 3
    max_num_nodes = 200


class SynManager(GraphDatasetManager):
    
    def _process(self):
        pass
    def _download(self):
        pass
    
class SynCC(SynManager):
    name = "syn_cc_0.9"
    _fea_dim = 1
    _target_dim = 10
    max_num_nodes = 100
  
class SynCC(SynManager):
    name = "syn_cc_0.9"
    _fea_dim = 1
    _target_dim = 10
    max_num_nodes = 100
    
class SynCC(SynManager):
    name = "syn_cc_0.9"
    _fea_dim = 1
    _target_dim = 10
    max_num_nodes = 100
    
class SynCC(SynManager):
    name = "syn_cc"
    corr = "0.9"
    _fea_dim = 1
    _target_dim = 10
    max_num_nodes = 100

class SynDegree(SynManager):
    name = "syn_degree"
    corr = "0.9_class10"
    _fea_dim = 1
    _target_dim = 10
    max_num_nodes = 100
    