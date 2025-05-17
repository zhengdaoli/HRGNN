import sys,os
sys.path.append(os.getcwd())

import argparse
from EndToEnd_Evaluation import main as endtoend
from PrepareDatasets import DATASETS
from config.base import Grid, Config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l1_reg', type=float, default=0.01, help='l1_reg')
    parser.add_argument('--l2_reg', type=float, default=0.01, help='l2_reg')
    parser.add_argument('--feature_reg', type=float, default=0.01, help='feature_reg')
    parser.add_argument('--ggn_gnn_type', type=str, default='gcn', choices=['gcn', 'gin', 'gat'], help='GNN type in HGG')
    parser.add_argument('--conv_type', type=str, default='gcn', choices=['GCN', 'GIN', 'GAT'], help='GNN type in graph HGG ENCoder')
    # ori_ratio
    parser.add_argument('--ori_ratio', type=float, default=0.5, help='ori_ratio')
    parser.add_argument('--aug_ratio', type=float, default=0.5, help='aug_ratio')
    parser.add_argument('--k_cross', type=int, default=1, help='k_cross')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--patience_epochs', type=int, default=100, help='patience_epochs')
    
    parser.add_argument('--gat_heads', type=int, default=8, help='heads number of GAT')

    parser.add_argument('--perturb_type', type=str, default='random', help='perturb_type')

    parser.add_argument('--use_hvo', type=str, default='False', help='use_hvo')
    parser.add_argument('--k_components', type=int, default=1, help='Number of components of mixture Gassuaians')
    parser.add_argument('--epochs', type=int, default=400, help='epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lrG', type=float, default=0.001, help='learning rate for Generative')
    parser.add_argument('--lrP', type=float, default=0.01, help='learning rate for prediction')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='lr_decay_rate')
    parser.add_argument('--inner_processes_G', dest='inner_processes_G', type=int, default=1)
    parser.add_argument('--inner_processes_F', dest='inner_processes_F', type=int, default=1)
    parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
    parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
    parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
    parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
    parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
    parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
    parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')

    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--experiment', dest='experiment', default='endtoend')
    parser.add_argument('--result-folder', dest='result_folder', default='RESULTS')
    parser.add_argument('--dataset-name', dest='dataset_name', default='none')
    parser.add_argument('--dataset_para', dest='dataset_para',type=str, default='0.9')
    parser.add_argument('--outer_folds', dest='outer_folds', default=10)
    parser.add_argument('--rewire_ratio', dest='rewire_ratio', type=str, default='0.0')
    parser.add_argument('--outer-processes', dest='outer_processes', default=2)
    parser.add_argument('--inner-folds', dest='inner_folds', default=5)
    parser.add_argument('--inner-processes', dest='inner_processes', default=1)
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
    parser.add_argument('--gen_type', type=str, dest='gen_type', choices=['gsample','node_hgg', 'graph_vgae','node_vgae','graph_hgg','vgae', 'mock'], default=None)

    
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print('args dict: ----  ', args.__dict__)
    if args.dataset_name not in ['all', 'none']:
        datasets = [args.dataset_name]
    else:
        datasets = list(DATASETS.keys())
        
        # ['IMDB-MULTI', 'IMDB-BINARY', 'PROTEINS', 'NCI1', 'ENZYMES', 'DD',
                    # 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB', 'REDDIT-MULTI-12K']

    config_file = args.config_file
    experiment = args.experiment
    
    for dataset_name in datasets:
        try:
            model_configurations = Grid(config_file, dataset_name)
            # NOTE: override value from args.
            model_configurations.override_by_dict(args.__dict__)
            
            endtoend(model_configurations,
                     outer_k=int(args.outer_folds), outer_processes=int(args.outer_processes),
                     inner_k=int(args.inner_folds), inner_processes=int(args.inner_processes),
                     result_folder=args.result_folder, debug=args.debug, repeat=args.repeat, pretrain=args.pretrain)
        
        except Exception as e:
            raise e  # print(e)