import sys, os
sys.path.append(os.getcwd())

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import torch_geometric.utils as torch_utils

from torch import nn
import time
from node_task import utils
from node_task.utils import normalize_adj
from models.generative_models import GCN, DGGN, DenseGCN, GNNPredictor, OriGCN
from betagnn.models import CombinedModel, NodeFeatureModel
from betagnn.models import GCN as betagnn_GCN


_OUTER_RESULTS_FILENAME = 'outer_results.json'
_ASSESSMENT_FILENAME = 'assessment_results.json'


def select_model(model_name, num_features, num_classes, args, adj=None):
    if model_name == 'GCN':
        # gnn = DenseGCN(num_features, num_classes, hidden_dims=[16, 16], dropout=args.dropout)
        # model = GNNPredictor(gnn)
        # nfeat, nhid, nclass, dropout):
        model = OriGCN(num_features, 16, num_classes, dropout=args.dropout)
    elif model_name == 'DGGN':
        model = DGGN(num_features, 0, num_classes, config=vars(args), adj=adj, args=args)
        
    elif model_name == 'LearnGCN':
        # learnable matrix with adj as initialization:
        learnable_A = torch.nn.Parameter(adj)
        model = OriGCN(num_features, 16, num_classes, dropout=args.dropout, adj=learnable_A)
    elif model_name == 'betaGNN':
        # in_channels, hidden_channels, out_channels, dropout_rate):
        gcn = betagnn_GCN(in_channels=num_features, hidden_channels=16, 
                          out_channels=num_classes, dropout_rate=args.dropout)
        # gcn_model, node_feature_model, out_channels):
        feat_model = NodeFeatureModel(in_channels=num_features, hidden_channels=16,
                                      out_channels=num_classes, dropout_rate=args.dropout)
        model = CombinedModel(gcn_model=gcn, node_feature_model=feat_model, out_channels=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes, hidden_dims=[16], dropout=0.3):
#         super(GCN, self).__init__()
        
#         # Create a list of GCNConv layers based on the hidden_dims
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GCNConv(num_features, hidden_dims[0]))
#         for i in range(1, len(hidden_dims)):
#             self.convs.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
#         self.convs.append(GCNConv(hidden_dims[-1], num_classes))
#         self.save_features = False
#         self.dropout = dropout

#     def forward(self, data):
#             x, edge_index = data.x, data.edge_index
            
#             # Pass through the GCNConv layers
#             for i, conv in enumerate(self.convs):
#                 x = conv(x, edge_index)
#                 if i != len(self.convs) - 1:  # No activation & dropout on the last layer
#                     x = F.relu(x)
#                     x = F.dropout(x, p=self.dropout, training=self.training)
            
#             if self.save_features:
#                 self.features = x.detach()  # Save the features
            
#             return F.log_softmax(x, dim=1)


import numpy as np
import matplotlib.cm as cm

def visualize_with_tsne(features, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    embedded_features = tsne.fit_transform(features)

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = cm.rainbow(np.linspace(0, 1, num_classes))

    plt.figure(figsize=(10, 10), dpi=150)
    
    for i, label in enumerate(unique_labels):
        plt.scatter(embedded_features[labels == label, 0],
                    embedded_features[labels == label, 1],
                    color=colors[i],
                    s=5,
                    label=f'Class {label}')
    # make the legend font larger:
    
    plt.title(f't-SNE visualization of {save_path.split("_")[-3]} features')
    plt.legend(prop={'size':20})
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_dataset(name='Cora'):
    # Load the dataset
    # `"CiteSeer"`, :obj:`"PubMed"`).
    dataset = Planetoid(root='DATA', name=name)
    return dataset

import matplotlib.pyplot as plt

def plot_loss_acc(results, save_path):
    train_losses = results['train_loss']
    val_losses = results['val_loss']
    train_accs = results['train_acc']
    val_accs = results['val_acc']

    fig, ax1 = plt.subplots(dpi=150)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.plot(train_losses, label='Train')
    ax1.legend()
    
    ax2 = ax1.twinx()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Train Accuracy')
    ax2.plot(train_accs, label='Train')
    ax2.legend()

    plt.savefig(save_path + '_train.png')
    plt.close()

    fig, ax1 = plt.subplots(dpi=150)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.plot(val_losses, label='Validation')
    ax1.legend()
    
    ax2 = ax1.twinx()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.plot(val_accs, label='Validation')
    ax2.legend()

    plt.savefig(save_path + '_val.png')
    plt.close()
    
def test_model(args, trainer, data, json_outer_results, outer_k, pretrain=False):
    model.save_features = True
    from collections import defaultdict
    all_results = defaultdict(list)
    # not use validation best.
    model.load_state_dict(torch.load(args.best_model_save_path))
    trainer.model = model

    # print the pytorch model structure:
    print('model 2:', model.state_dict())

    # save and print the inference time:
    import time
    start_time = time.time()
    test_results = trainer.test(data)
    elp_time = time.time() - start_time
    print(f'Inference time: {elp_time:.4f}')
    # TSNE:
    # features = model.features.cpu().numpy()
    # labels = data.y.cpu().numpy()
    # visualize_with_tsne(features, labels, args.tsne_save_path)
    
    # if args.model_name == 'DGGN':
    #     aug_A = trainer.model.generate_graph(data, data.x, data.edge_index)
    #     # save the heatmap of aug_A, with bar:
    #     plt.figure(figsize=(10, 10), dpi=150)
    #     plt.imshow(aug_A.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    #     plt.colorbar()
    
    #     plt.savefig(args.tsne_save_path.replace('.png', '_heatmap.png'))
    
    print(f'Test Accuracy: {test_results["acc"]:.4f}')
    all_results['test_acc'] = test_results['acc']
    print('all_results:', all_results)



def run_model(args, trainer, data, json_outer_results, outer_k, pretrain=False):
    best_val_acc = 0
    patience_counter = 0
    # print all classes:
    print('all classes:', np.unique(data.y.cpu().numpy()))
    
    from collections import defaultdict

    all_results = defaultdict(list)

    
    for epoch in range(1, args.epochs + 1):
        results = trainer.train(data)
        val_results = trainer.eval(data)
        train_out_str = ''
        for key, value in results.items():
            train_out_str += f'Train {key}: {value:.4f}, '
        train_out_str = train_out_str.strip()
        
        print(f'Epoch: {epoch:03d}, {train_out_str}, Val Acc: {val_results["acc"]:.4f}')
        
        all_results['val_acc'].append(val_results['acc'])
        all_results['val_loss'].append(val_results['loss'])
        all_results['train_acc'].append(results['acc'])
        # all_results['train_loss'].append(results['loss'])

        # Early stopping based on validation accuracy
        if epoch > 10 and val_results["acc"] > best_val_acc:
            best_val_acc = val_results["acc"]
            patience_counter = 0
            # Save the best model
            torch.save(trainer.model.state_dict(), args.best_model_save_path)
        else:
            if epoch > 10:
                patience_counter += 1
            
            if patience_counter >= args.patience_epochs:
                print("Early stopping triggered.", 'best epoch:', epoch)
                break
            
    plot_loss_acc(all_results, args.tsne_save_path.replace('.png', '_acc_loss'))
    # Load the best model for testing
    model.save_features = True

    # not use validation best.
    
    model.load_state_dict(torch.load(args.best_model_save_path))
    trainer.model = model

    test_results = trainer.test(data)
    
    # TSNE:
    # features = model.features.cpu().numpy()
    # labels = data.y.cpu().numpy()
    # visualize_with_tsne(features, labels, args.tsne_save_path)
    
    # if args.model_name == 'DGGN':
    #     aug_A = trainer.model.generate_graph(data, data.x, data.edge_index)
    #     # save the heatmap of aug_A, with bar:
    #     plt.figure(figsize=(10, 10), dpi=150)
    #     plt.imshow(aug_A.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    #     plt.colorbar()
    
    #     plt.savefig(args.tsne_save_path.replace('.png', '_heatmap.png'))
    
    print(f'Test Accuracy: {test_results["acc"]:.4f}')
    all_results['test_acc'] = test_results['acc']
    # save all_results as json:
    import json
    with open(json_outer_results, 'w') as f:
        json.dump(all_results, f)
    print(f'Saved results to {json_outer_results}')
    

def get_args():
    parser = argparse.ArgumentParser()
    # add these parameters: self.args.l1_reg, self.args.l2_reg, self.args.feature_reg
    parser.add_argument('--l1_reg', type=float, default=0.01, help='l1_reg')
    parser.add_argument('--l2_reg', type=float, default=0.01, help='l2_reg')
    parser.add_argument('--feature_reg', type=float, default=0.01, help='feature_reg')
    parser.add_argument('--ggn_gnn_type', type=str, default='gcn', choices=['gcn', 'gin', 'gat'], help='GNN type in HGG')
    
    parser.add_argument('--gat_heads', type=int, default=8, help='heads number of GAT')

    parser.add_argument('--perturb_type', type=str, default='random', help='perturb_type')
    parser.add_argument('--tsne_save_path', type=str, default='./tsne_visualization.png', help='Path to save the t-SNE visualization')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--task_type', type=str, default='node', help='task_type')
    parser.add_argument('--dev', action='store_true', help='dev')
    parser.add_argument('--use_hvo', type=str, default='False', help='use_hvo')
    parser.add_argument('--k_components', type=int, default=1, help='Number of components of mixture Gassuaians')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--ori_ratio', type=float, default=1.0, help='ori adj ratio')
    parser.add_argument('--aug_ratio', type=float, default=0.0, help='aug adj ratio')
    parser.add_argument('--patience_epochs', type=int, default=10, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[16], help='List of hidden dimensions for GCN layers')
    parser.add_argument('--grad_norm', action='store_true', help='grad norm')
    parser.add_argument('--max_grad_norm', type=int, default=5, help='max_grad_norm')
    parser.add_argument('--k_cross', type=int, default=1, help='k_cross')
    parser.add_argument('--model_name', type=str, default='GCN', help='model_name')
    parser.add_argument('--exp_path', type=str, default='./', help='exp_path')
    parser.add_argument('--load_data', action='store_true', help='only load_data')
    parser.add_argument('--reglow', action='store_true', help='regularization low rank')
    parser.add_argument('--dev_size', type=int, default=1000, help='dev_sample_size')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device, e.g., cuda:0')
    parser.add_argument('--inference', type=str, default='False', help='inference')
    parser.add_argument('--Lambda', type=float, default=0.5, help='lambda')
    parser.add_argument('--best_model_save_path', type=str, default='.best_model', help='best_model')
    parser.add_argument('--pre_model_path', type=str, default='./pre_model/best_model', help='pre_model_path')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=400, help='epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lrG', type=float, default=0.001, help='learning rate for Generative')
    parser.add_argument('--lrP', type=float, default=0.01, help='learning rate for prediction')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='lr_decay_rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--clip', type=int, default=3, help='clip')
    parser.add_argument('--seq_length', type=int, default=12, help='seq_length')
    parser.add_argument('--predict_len', type=int, default=12, help='predict_len')
    parser.add_argument('--scheduler', action='store_true', help='scheduler')
    parser.add_argument('--config_file', dest='config_file')
    parser.add_argument('--experiment', dest='experiment', default='endtoend')
    parser.add_argument('--result_folder', dest='result_folder', default='RESULTS')
    parser.add_argument('--dataset_name', dest='dataset_name', default='Cora')
    parser.add_argument('--dataset_para', dest='dataset_para',type=str, default='0.9')
    parser.add_argument('--outer_folds', dest='outer_folds', default=10)
    parser.add_argument('--rewire_ratio', dest='rewire_ratio', type=float, default=0.001)
    parser.add_argument('--outer_processes', dest='outer_processes', default=2)
    parser.add_argument('--inner_folds', dest='inner_folds', default=5)
    parser.add_argument('--inner_processes_G', dest='inner_processes_G', type=int, default=1)
    parser.add_argument('--inner_processes_F', dest='inner_processes_F', type=int, default=1)
    parser.add_argument('--debug', action="store_true", dest='debug')
    parser.add_argument('--mol_split', type=bool, dest='mol_split', default=False)
    parser.add_argument('--repeat', type=bool, dest='repeat', default=False)
    parser.add_argument('--pretrain', type=bool, dest='pretrain', default=False)
    parser.add_argument('--ogb_evl', type=bool, dest='ogb_evl', default=False)
    parser.add_argument('--wandb', type=bool, dest='wandb', default=False)
    parser.add_argument('--pretrain_model_date', type=str, dest='pretrain_model_date', default=None)
    parser.add_argument('--pretrain_model_folder', type=str, dest='pretrain_model_folder', default=None)
    parser.add_argument('--model_checkpoint_path', type=str, dest='model_checkpoint_path', default='/li_zhengdao/github/GenerativeGNN/rewiring_models')
    parser.add_argument('--perturb_op', type=str, dest='perturb_op', choices=['rewire', 'drop', 'add'], default=None)
    parser.add_argument('--job_type_name', type=str, dest='job_type_name', default='default')
    parser.add_argument('--gen_type', type=str, dest='gen_type', choices=['gsample','node_hgg','node_vgae', 'vgae', 'mock'], default=None)
    parser.add_argument('--hidden', type=int, default=16,help='Number of hidden units.')
    parser.add_argument('--attack', type=str, default='meta', choices=['no', 'meta', 'random', 'nettack'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
    parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
    parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
    parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
    parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
    parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
    parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
    parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
    parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
    parser.add_argument('--symmetric', action='store_true', default=False, help='whether use symmetric matrix')
                        
                    
    return parser


def run_pro_gnn(args, dataset, perturbed_adj, num_fea, num_classes,json_outer_results, outer_k):
    
    from models.pro_gnn import ProGNN
    
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    model = OriGCN(num_fea, 16, num_classes, dropout=args.dropout)
    prognn = ProGNN(model, args, args.device)
    
    prognn.fit(dataset.x, perturbed_adj, dataset.y, dataset.train_mask, dataset.val_mask)
    acc_test = prognn.test(dataset.x, dataset.y, dataset.test_mask)
    results = {'test_acc': acc_test}
    import json
    with open(json_outer_results, 'w') as f:
        json.dump(results, f)
    print(f'Saved results to {json_outer_results}')


if __name__=='__main__':
    
    args = get_args()
    args = args.parse_args()
    
    print(f'Starting experiment with args: {args}')
    time.sleep(3)

    # Initialize the model and optimizer
    device = args.device
    print('cuda device:', device)
    
    dataset = load_dataset(args.dataset_name)
    data = dataset[0].to(device)
    
    adj = torch_utils.to_dense_adj(data.edge_index)[0]
    print('adj shape:', adj.shape)
    
    print('rewire_ratio:', args.rewire_ratio)
    print(args.rewire_ratio > 0.000001)
    
    if args.perturb_type == 'random' and args.rewire_ratio > 0.000001:
        from deeprobust.graph.global_attack import Random
        from my_utils import numpy_to_csr
        
        # import random; random.seed(args.seed)
        # np.random.seed(args.seed)
        attacker = Random()
        # transform adj to scipy.sparse.csr_matrix
        ori_adj = adj
        adj = numpy_to_csr(adj.cpu().numpy())
        n_perturbations = int(args.rewire_ratio * (adj.sum()//2))
        attacker.attack(adj, n_perturbations, type='add')
        perturbed_adj = attacker.modified_adj
        # transform perturbed_adj to torch tensor:
        adj = torch.from_numpy(perturbed_adj.todense()).float().to(device)
        print('perturbed adj shape:', adj.shape)
        # count different between adj and ori_adj:
        diff = (adj - ori_adj).sum()
        print('diff:', diff)
    
    # TODO: to dense adj:
    from node_task.preprocessing import *

    if args.model_name != 'pro_gnn' and args.ggn_gnn_type == 'gcn':
        norm_adj = normalize_adj(adj)
    else:
        norm_adj = adj
    from torch_geometric.utils import to_edge_index
    from torch_geometric.utils import dense_to_sparse

    edge_index, edge_attr = dense_to_sparse(adj)
    data.edge_index = edge_index

    print('num classes:', dataset.num_classes)
    print('num features:', dataset.num_features)
    print('adj[0]: ', adj[0])
    

    torch.autograd.set_detect_anomaly(True)
    
    k_cross = args.k_cross
    exp_path = args.exp_path
    
    __NESTED_FOLDER = os.path.join(exp_path, f'{k_cross}_NESTED_CV')
    
    for outer_k in range(1, k_cross+1):
        # Create a separate folder for each experiment
        if not os.path.exists(__NESTED_FOLDER):
            os.makedirs(__NESTED_FOLDER)
        json_outer_results = os.path.join(__NESTED_FOLDER, f'{outer_k}_{_OUTER_RESULTS_FILENAME}')
        if not os.path.exists(json_outer_results) or args.inference == 'True':
            if args.model_name == 'pro_gnn':
                print('run pro_gnn...')
                run_pro_gnn(args, data, adj, dataset.num_features, dataset.num_classes,json_outer_results, outer_k)
            else:
                model = select_model(args.model_name, dataset.num_features, dataset.num_classes, args, adj=adj).to(device)
                print('model:', model.state_dict())
                load_pretrain = False
                if load_pretrain:
                    model.graph_gen.load_state_dict(torch.load('./vgae_model.pth'))
                    print('load graph_gen statedict from:', args.pre_model_path)
                if args.model_name == 'GCN':
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    criterion = nn.CrossEntropyLoss().to(device)
                    trainer = utils.Trainer(args, model, optimizer, criterion)
                elif args.model_name == 'LearnGCN':
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    criterion = nn.CrossEntropyLoss().to(device)
                    trainer = utils.Trainer(args, model, optimizer, criterion)
                elif args.model_name == 'DGGN':
                    optimizer_G= torch.optim.Adam(model.graph_gen.parameters(), lr=args.lrG, weight_decay=1e-5)
                    optimizer_F = torch.optim.Adam(model.gnn.parameters(), lr=args.lrP, weight_decay=args.weight_decay)
                    # combine to paras graph_gen and gnn, and use one optimizer to opt both paras:
                    
                    # optimizer = torch.optim.Adam([
                    #     {'params': model.graph_gen.parameters(), 'lr': args.lr},
                    #     {'params': model.gnn.parameters(), 'lr': args.lr}
                    # ], weight_decay=args.weight_decay)
                    # trainer = utils.GGNTrainer(args, model, optimizer=optimizer, criterion=F.nll_loss)
                    trainer = utils.GGNTrainer(args, model, optimizer_G=optimizer_G, optimizer_F=optimizer_F, criterion=F.nll_loss)
                elif args.model_name == 'betaGNN':
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    criterion = nn.CrossEntropyLoss().to(device)
                    trainer = utils.Trainer(args, model, optimizer, criterion)
                if args.inference == 'False':
                    run_model(args, trainer, data, json_outer_results, outer_k, args.pretrain)
                elif args.inference == 'True':
                    test_model(args, trainer, data, json_outer_results, outer_k, args.pretrain)
        else:
            # Do not recompute experiments for this outer fold.
            print(f"File {json_outer_results} already present! Shutting down to prevent loss of previous experiments")