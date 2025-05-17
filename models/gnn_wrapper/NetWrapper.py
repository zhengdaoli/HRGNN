import time
from datetime import timedelta
import torch
from torch import optim
from sklearn.metrics import roc_auc_score
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch import nn
from torch_geometric.data import DataLoader
from models.utils.EarlyStopper import ClassificationMetrics
import wandb
import os
import re
from utils.utils import is_nan_inf, fill_nan_inf
from collections import defaultdict
from torch_geometric.utils import to_dense_adj
import argparse

def format_time(avg_time):
    avg_time = timedelta(seconds=avg_time)
    total_seconds = int(avg_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{str(avg_time.microseconds)[:3]}"


os.environ["WANDB_API_KEY"] = "d386112e5bddd7049b66293ae61397109c39fd27"


    


class NetWrapper:
    def __init__(self, model, loss_function, device='cpu', classification=True, config={}, exp_path=None, 
                 round=0, pretrain=False, outer_k=0, use_wandb=False, optimizer_G=None, optimizer_F=None):
        self.model = model
        self.round = round
        self.outer_k = outer_k
        self.use_wandb = use_wandb
        self.pretrain = pretrain
        self.loss_fun = loss_function
        self.exp_path = exp_path
        # self.loss_fun = nn.BCEWithLogitsLoss()
        self.device = torch.device(device)
        self.classification = classification
        self.config = config
        self.optimizer_G = optimizer_G
        self.optimizer_F = optimizer_F

        # TODO: if target_dims == 2 then use roc_auc
        self.roc_auc = self.config['roc_auc'] if 'roc_auc' in self.config else False
        self.dataset_name = self.config['dataset_name']
        if use_wandb:
            dt_time = time.strftime("%Y-%m-%d-%H")
            
            job_type_name = self.config['job_type_name']
            print('wandb jobtype: ', job_type_name)
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "DGGNN"),
                # append the date and time to the name of the run
                name =f"outer_{outer_k}",
                group=f'{self.dataset_name}_{self.model.__class__.__name__}',
                job_type=job_type_name, # run for each time, may hourly
            )
            wandb.config.update(self.config.__dict__, allow_val_change=True)
        else:
            self.use_wandb = False
        
        self.evaluator = None
        print('config:', config)
        if 'ogb_evl' in config:
            if config['ogb_evl']:
                self.evaluator = Evaluator(config['dataset_name'].replace('_',  '-'))
                print('!      loaded evaluator !')

        # trans config dict to argparser:

        
        self.roc_auc = (self.model.target_dim == 2) or (self.evaluator is not None and self.evaluator.eval_metric == 'rocauc')
        
        if pretrain:
            model_path = self.checkpoint_path()
            # 'results/result_0712_GIN_lzd_attr_NCI1_0.18/GIN_NCI1_assessment/' replace second result_0712 with pretrain_model_folder:
            # model_path = model_path.split('/')
            # model_path = /li_zhengdao/github/GenerativeGNN/rewiring_models/NCI1/results/result_0713_GIN_lzd_attr_NCI1_0.18/GIN_NCI1_assessment/10_NESTED_CV/OUTER_FOLD_1/round0_GIN_best_model.pth
            # use regex to replace the 'result_dddd_*/' in model_path with pretrain_model_folder
            model_path = re.sub(r'result_\d{4}_[^\/]*\/', self.config["pretrain_model_folder"]+'/', model_path)
            print('load model_path: ', model_path)
            if not os.path.exists(model_path):
                for round in range(3):
                    model_path = re.sub(r'round\d{1}', f'round{round}', model_path)
                    if  os.path.exists(model_path):
                        break
                print('load model using model:', model_path)
            self.model.load_state_dict(torch.load(model_path))
            

    def _cal_evl(self, data_loader, y_true, y_pred, acc_all, loss_all):
        
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        
        if self.evaluator is not None:
            if y_pred.numel() == y_true.numel():
                y_true = y_true.view(y_pred.shape)
                y_pred_argmax = y_pred
            elif self.evaluator.num_tasks == 1:
                y_pred_argmax = torch.argmax(y_pred, dim=-1).squeeze()
                y_pred_argmax = y_pred_argmax.view(y_true.shape)
            else:
                y_pred_argmax = y_pred
                
            y_true_numpy = np.float32(y_true.numpy())
            y_pred_argmax = np.float32(y_pred_argmax.numpy().astype(np.float32))
            y_true_numpy[y_true_numpy < 0] = np.nan
            
            input_dict = {"y_true": y_true_numpy, "y_pred": y_pred_argmax}
            evl_res = self.evaluator.eval(input_dict)
            
        if self.roc_auc:
            if self.evaluator is not None:
                auc_roc = evl_res['rocauc']
            else:
                if y_pred.shape[-1] > 1:
                    y_pred_argmax = torch.argmax(y_pred, dim=-1)
                y_true = y_true.view(y_pred_argmax.shape)
                
                auc_roc = roc_auc_score(y_true.numpy(), y_pred_argmax.numpy())
                
        if self.evaluator is not None:
            if self.evaluator.num_tasks == 1 and y_pred.shape[-1] > 1:
                y_pred_argmax = torch.argmax(y_pred, dim=-1)
            else:
                y_pred_argmax = y_pred > 0.5
        else:
            y_pred_argmax = torch.argmax(y_pred, dim=-1)
            
        y_pred_argmax = y_pred_argmax.squeeze()
        y_true = y_true.squeeze()
        # print('y_pred_argmax shape:', y_pred_argmax.shape)
        is_labeled = y_true == y_true
        # print('is_labeled shape:', is_labeled.shape)
        acc_all = 100. * (y_pred_argmax[is_labeled] == y_true[is_labeled]).sum().float() / y_true[is_labeled].size(0) # True/True
        acc_all = acc_all.cpu().numpy().item()
        
        if self.classification:
            if self.roc_auc:
                return acc_all, loss_all / len(data_loader.dataset), auc_roc
            else:
                return acc_all, loss_all / len(data_loader.dataset)
            # if self.roc_auc:
            #     return acc_all / len(data_loader.dataset), loss_all / len(data_loader.dataset), auc_roc
            # else:
            #     return acc_all / len(data_loader.dataset), loss_all / len(data_loader.dataset)
        else:
            return None, loss_all / len(data_loader.dataset)
    
    def _cal_loss_acc(self, data, pred, res=None):
        results = {}
        if data.y.shape == pred.shape:
            # NOTE: multi label and multi classification
            is_labeled = data.y == data.y
            
            loss, acc = self.loss_fun(pred.to(torch.float32)[is_labeled],
                                data.y.to(torch.float32)[is_labeled], res=res)
        else:
            loss, acc = self.loss_fun(pred.to(torch.float32), data.y.to(torch.float32), res=res)
        results = {'loss': loss, 'acc': acc}
        return results


    def _train(self, train_loader, optimizer, clipping=None, epoch=None):
        model = self.model.to(self.device)
        
        if self.pretrain:
            model.eval()
        else:
            model.train()

        loss_all = 0
        acc_all = 0

        y_true = []
        y_pred = []

        match_loss = 0
        class_loss = 0
        prior_loss = 0
        link_loss = 0
        
        for i, data in enumerate(train_loader):
            if 'g_x' not in data and (data.x.shape[0] == 1 or data.batch[-1] == 0):
                print('x shape:', data.x.shape)
                continue
                
            if i == len(train_loader):
                print('len train_loader:', len(train_loader))
                continue
            
            data = data.to(self.device)
            optimizer.zero_grad()
            
            if hasattr(model, 'before_forward'):
                model.before_forward(epoch)
                
            res = model(data)
            
            if isinstance(res, dict):
                pred = res['y']
            else:
                pred = res

            if self.classification:
                if data.y.shape == pred.shape:
                    # NOTE: multi label and multi classification
                    is_labeled = data.y == data.y
                    
                    loss, acc = self.loss_fun(pred.to(torch.float32)[is_labeled],
                                        data.y.to(torch.float32)[is_labeled], res=res)
                else:
                    loss, acc = self.loss_fun(pred.to(torch.float32), data.y.to(torch.float32), res=res)
                
                all_loss = 0
                if isinstance(loss, tuple):
                    for l in loss:
                        all_loss += l
                    ori_all_loss = all_loss.item()
                    
                    # check thess losses has inf:
                    
                    # print('ori_all_loss:', ori_all_loss, is_nan_inf(all_loss))
                    # print('class_loss:', loss[0], is_nan_inf(loss[0]))
                    # print('match_loss:', loss[1], is_nan_inf(loss[1]))
                    # print('prior_loss:', loss[2], is_nan_inf(loss[2]))
                    # print('link_loss:', loss[3], is_nan_inf(loss[3]))

                    # print("before all_loss nan:", is_nan_inf(all_loss), all_loss)
                    
                    all_loss.backward()
                    
                    # print("after all_loss nan:", is_nan_inf(all_loss))

                    class_loss += loss[0].item()
                    match_loss += loss[1].item()
                    prior_loss += loss[2].item()
                    link_loss += loss[3].item()
                    
                else:
                    loss.backward()
                    all_loss = loss
                try:
                    num_graphs = data.num_graphs
                except TypeError:
                    num_graphs = data.adj.size(0)
                    
                loss_all += all_loss.item() * num_graphs
                acc_all += acc.item() * num_graphs
            else:
                loss = self.loss_fun(res, data.y,res=res)
                loss.backward()
                loss_all += loss.item()
            
            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
                
            optimizer.step()
            
            # Calculate the gradient norm
            grad_norm = 0.0
            for param in model.parameters():
                if param is None or param.grad is None:
                    continue
                grad_norm += torch.norm(param.grad)**2
            grad_norm = grad_norm.sqrt().item()

            # print("Total Gradient Norm:", grad_norm)
            
            def print_grad_norm(cur_module, name):
                cur_grad_norm = 0.0
                for param in cur_module.parameters():
                    cur_grad_norm += torch.norm(param.grad)**2
                cur_grad_norm = cur_grad_norm.sqrt().item()
                print(f"{name} Gradient Norm:", cur_grad_norm)

            # print_grad_norm(self.model.vgae.base_gcn.first_h, 'first_h')
            # print_grad_norm(self.model.vgae.base_gcn.linears, 'linears')
            # print_grad_norm(self.model.vgae.base_gcn.convs, 'convs')
            # print_grad_norm(self.model.vgae.base_gcn.nns, 'nns')
            
            # print_grad_norm(self.model.vgae.gcn_mean, 'gcn_mean')
            
            # print_grad_norm(self.model.vgae.gcn_logstddev, 'gcn_logstddev')
            
            # if self.evaluator is not None or self.roc_auc:
            y_true.append(data.y.detach().cpu())
            y_pred.append(pred.detach().cpu())
            
        return self._cal_evl(train_loader, y_true, y_pred, acc_all, loss_all), (class_loss, match_loss, prior_loss, link_loss)

    def classify_graphs(self, loader):
        model = self.model.to(self.device)
        model.eval()

        loss_all = 0
        acc_all = 0
        y_true = []
        y_pred = []
        
        class_loss = 0
        match_loss = 0
        prior_loss = 0
        link_loss = 0
        
        for data in loader:
            data = data.to(self.device)
            res = model(data)
                        
            if isinstance(res, dict):
                pred = res['y']
            else:
                pred = res

            if self.classification:
                if data.y.shape == pred.shape:
                    # NOTE: multi label and multi classification
                    is_labeled = data.y == data.y
                    loss, acc = self.loss_fun(pred.to(torch.float32)[is_labeled],
                                        data.y.to(torch.float32)[is_labeled])
                else:
                    float_targets = data.y.to(torch.float32)
                    float_pred = pred.to(torch.float32)
                    
                    loss, acc = self.loss_fun(float_pred, float_targets, res=res)
                try:
                    num_graphs = data.num_graphs
                except TypeError:
                    print('no num_graphs')
                    num_graphs = data.adj.size(0)
                    
                if isinstance(loss, tuple):
                    class_loss += loss[0].item()
                    match_loss += loss[1].item()
                    prior_loss += loss[2].item()
                    link_loss += loss[3].item()
                    
                    loss_all = class_loss + match_loss + prior_loss + link_loss
                    
                else:
                    loss_all += loss.item() * num_graphs
                acc_all += acc.item() * num_graphs
            else:
                loss = self.loss_fun(pred, data.y, res=res)
                loss_all += loss.item()
                
            # if self.evaluator is not None or self.roc_auc:
            y_true.append(data.y.detach().cpu())
            y_pred.append(pred.detach().cpu())

        return self._cal_evl(loader, y_true, y_pred, acc_all, loss_all), (class_loss, match_loss, prior_loss, link_loss)
    

    def checkpoint_path(self):
        model_checkpoint_path = self.config['model_checkpoint_path']
        root_path = os.path.join(model_checkpoint_path, self.config['dataset_name'])
        # create fold if not exist:
        save_path = os.path.join(root_path, self.exp_path)
                
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, f'round{self.round}_{self.model.__class__.__name__}_best_model.pth')
        return path
    
    def save_model(self):
            # save model at that fold:
        model_checkpoint_path = self.checkpoint_path()
        torch.save(self.model.state_dict(), model_checkpoint_path)
        """
         /li_zhengdao/github/GenerativeGNN/rewiring_models/NCI1/results/result_0712_GIN_lzd_attr_NCI1_0.02/GIN_NCI1_assessment/10_NESTED_CV/OUTER_FOLD_1/round0_GIN_best_model.pth
        """
        print('saved model to ', model_checkpoint_path)
        
    # def load_model(self):
    #     # load model:
    #     model = 
        
    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam,
              scheduler=None, clipping=None,
              validation_loader=None, 
              test_loader=None, early_stopping=None, 
              logger=None, log_every=1, need_save_model=False) -> ClassificationMetrics:

        early_stopper = early_stopping()
        

        val_loss, val_acc = -1, -1
        test_loss, test_acc = None, None
        train_roc_auc, val_roc_auc, test_roc_auc = -1, -1, -1
        train_acc, train_loss, train_roc_auc, class_loss, match_loss,total_norm = -1, -1, -1, -1, -1, -1

        time_per_epoch = []

        for epoch in range(1, max_epochs+1):
            start = time.time()
            
            if not self.pretrain:
                """training if not pretrain.
                """
                if self.roc_auc:
                    (train_acc, train_loss, train_roc_auc), loss_each = self._train(train_loader, optimizer, clipping, epoch=epoch)
                else:
                    (train_acc, train_loss), loss_each = self._train(train_loader, optimizer, clipping, epoch=epoch)

                # TODO: calculate norm before clipping 

                # total_norm = 0.0
                # for p in self.model.parameters():
                #     if p.grad is None:
                #         param_norm = p.norm(2)
                #     else:
                #         param_norm = p.grad.data.norm(2)
                        
                #     total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** (1. / 2)

                if scheduler is not None:
                    scheduler.step()
                    
            end = time.time() - start
            time_per_epoch.append(end)

            if test_loader is not None:
                inference_time = time.time()
                if self.roc_auc:
                    (test_acc, test_loss, test_roc_auc), loss_each = self.classify_graphs(test_loader)
                else:
                    (test_acc, test_loss), loss_each = self.classify_graphs(test_loader)
                print('inference time:', time.time() - inference_time)

            if validation_loader is not None:
                if self.roc_auc:
                    (val_acc, val_loss, val_roc_auc), loss_each = self.classify_graphs(validation_loader)
                else:
                    (val_acc, val_loss), loss_each = self.classify_graphs(validation_loader)

                # Early stopping (lazy if evaluation)
                if self.roc_auc:
                    if early_stopper.stop(epoch, val_loss, val_acc,
                                                test_loss, test_acc,
                                                train_loss, train_acc,
                                                train_roc_auc,
                                                val_roc_auc,
                                                test_roc_auc):
                        msg = f'Stopping at epoch {epoch}, best is {early_stopper.get_best_vl_metrics().__dict__}'
                        if logger is not None:
                            logger.log(msg)
                            print(msg)
                        else:
                            print(msg)
                        break
                else:
                    if early_stopper.stop(epoch, val_loss, val_acc,
                                                test_loss, test_acc,
                                                train_loss, train_acc):
                        msg = f'Stopping at epoch {epoch}, best is {early_stopper.get_best_vl_metrics().__dict__}'
                    
                        if logger is not None:
                            logger.log(msg)
                            print(msg)
                        else:
                            print(msg)
                        break
            
            if not self.pretrain and need_save_model and early_stopper.need_save():
                self.save_model()
                
            class_loss, match_loss, prior_loss, link_loss = loss_each
            if self.use_wandb:
                
                wandb.log({'dataset_name': self.dataset_name, 'Epoch': epoch, 'Train Loss': train_loss, 'Train Acc': train_acc,
                       'Val Loss': val_loss, 'Val Acc': val_acc, 'Train roc_auc': train_roc_auc,
                       'Val roc_auc': val_roc_auc, 'Test Loss': test_loss, 'Test Acc': test_acc,
                       'Test roc_auc': test_roc_auc, 'class_loss': class_loss, 
                       'match_loss': match_loss, 'prior_loss': prior_loss, 'link_loss': link_loss})
            
            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR acc: {train_acc}, VL loss: {val_loss} VL acc: {val_acc} ' \
                    f'TE loss: {test_loss} TE acc: {test_acc}, GradNorm: {total_norm}, TR rocauc: {train_roc_auc}, VL rocauc:{val_roc_auc}, \
                        TE rocauc: {test_roc_auc}, link_loss: {link_loss}, prior_loss: {prior_loss}, \
                        class_loss: {class_loss}, match_loss: {match_loss}'
                    # TODO: add grad norm
                    
                if logger is not None:
                    logger.log(msg)
                else:
                    print(msg)
                    
            if self.pretrain:
                # first epoch is the best epoch
                print('break pretrain, epoch: ', epoch)
                break
            
        wandb.finish()
        time_per_epoch = torch.tensor(time_per_epoch)
        avg_time_per_epoch = float(time_per_epoch.mean())

        elapsed = format_time(avg_time_per_epoch)

        print('elapsed per epoch: ', elapsed)
        
        return early_stopper.get_best_vl_metrics()
    
# Since PyTorch Geometric is not available, we will create a custom function to convert edge indices to dense adjacency matrices.

def edge_index_to_dense(edge_index, num_nodes):
    # Create an empty adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    # Fill in the adjacency matrix
    adj_matrix[edge_index[0], edge_index[1]] = 1
    return adj_matrix

def custom_isin(elements, test_elements):
    return (elements[..., None] == test_elements).any(-1)



def convert_to_dense_adj_custom(edge_index, batch):
    dense_adj_matrices = []
    
    for i in range(batch.max().item() + 1):
        # Get the node indices for the current graph
        node_indices = (batch == i).nonzero(as_tuple=False).view(-1)

        # Find the edges for the current graph
        mask = custom_isin(edge_index[0], node_indices) & custom_isin(edge_index[1], node_indices)
        subgraph_edge_index = edge_index[:, mask]

        # Adjust the edge indices to be local to the current graph
        subgraph_edge_index = subgraph_edge_index - node_indices.min()

        # Convert the subgraph's edge index to a dense adjacency matrix
        num_nodes = node_indices.unique().size(0)
        dense_adj_matrix = edge_index_to_dense(subgraph_edge_index, num_nodes)

        dense_adj_matrices.append(dense_adj_matrix.to(edge_index.device))

    return dense_adj_matrices




class DGGNNetWrapper(NetWrapper):
    def __init__(self, model, loss_function, device='cpu', classification=True, config={}, exp_path=None, round=0, pretrain=False, outer_k=0, use_wandb=False):
        super(DGGNNetWrapper, self).__init__(model, loss_function, device, classification, config, exp_path, round, pretrain, outer_k, use_wandb)
        self.model = model
        self.round = round
        self.outer_k = outer_k
        self.use_wandb = use_wandb
        self.pretrain = pretrain
        self.loss_fun = loss_function
        self.exp_path = exp_path
        # self.loss_fun = nn.BCEWithLogitsLoss()
        self.device = torch.device(device)
        self.classification = classification
        self.config = config
        self.args = argparse.Namespace(**config.config)

        if self.optimizer_G is None:
            self.optimizer_G= torch.optim.Adam(model.graph_gen.parameters(), lr=self.args.lrG, weight_decay=1e-5)
            self.optimizer_F = torch.optim.Adam(model.gnn.parameters(), lr=self.args.lrP, weight_decay=self.args.weight_decay)


    def classify_graphs(self, loader):

        loss_all = 0
        acc_all = 0
        y_true = []
        y_pred = []
        
        all_results = {}
        for i, data in enumerate(loader):

            data = data.to(self.device)
            ori_As = convert_to_dense_adj_custom(data.edge_index, data.batch)
            preddictor_results = self.train_predictor(data, ori_As)

            pred = preddictor_results['pred']

            loss = preddictor_results['loss']
            acc = preddictor_results['acc']

            try:
                num_graphs = data.num_graphs
            except TypeError:
                num_graphs = data.adj.size(0)

            loss_all += loss * num_graphs
            acc_all += acc * num_graphs

            for k, v in preddictor_results.items():
                all_results[k] = v
                
            # if self.evaluator is not None or self.roc_auc:
            y_true.append(data.y.detach().cpu())
            y_pred.append(pred.detach().cpu())

        return self._cal_evl(loader, y_true, y_pred, acc_all, loss_all), all_results


    def train_adj(self, data, ori_As):
        # NOTE: in one batch:
        x, edge_index = data.x, data.edge_index
        batch = data.batch

        self.model.graph_gen.train()
        self.optimizer_G.zero_grad()
        
        aug_edeg_index, aug_dense_As = self.model.generate_graph(data, x, edge_index)

        # check all the shape whether same in ori_As and aug_dense_As:
        print('len ori_As:', len(ori_As))
        loss_prior = self.model.update_prior_graph(aug_dense_As, ori_As)
        loss_re = self.model.loss_re(aug_dense_As, ori_As)

        edge_index = self.model.modify_aug_A(aug_dense_As, ori_As, batch)

        self.model.gnn.eval()

        # sce loss:
        # with torch.no_grad():
        #     _ = self.model.graph_predict(data, x=x, edge_index=data.edge_index)
        #     ori_features = self.model.features
        #     _ = self.model.graph_predict(data, x=x, edge_index=edge_index)
        #     aug_features = self.model.features

        # loss_fea_re = self.model.sce_loss(aug_features, ori_features)

        loss_all = loss_re + loss_prior

        # loss_all = loss_re
        loss_all.backward()
        self.optimizer_G.step()

        with torch.no_grad():
            outputs = self.model.graph_predict(data, x=x, edge_index=edge_index)
            pred = outputs['y']
            print('pred shape;', pred.shape)
            results = self._cal_loss_acc(data, pred, res=None)
            loss = results['loss']
        
        
        results['loss_re'] = loss_re.item()
        results['loss_pred_adj'] = loss.item()
        results['loss_prior'] = loss_prior.item()

        # results['loss_fea_re'] = loss_fea_re.item()
        
        return results
    
    def train_predictor(self, data, ori_As):
        x, edeg_index = data.x, data.edge_index
        batch = data.batch

        self.model.gnn.train()
        self.optimizer_F.zero_grad()

        self.model.graph_gen.eval()

        aug_edeg_index, aug_dense_As = self.model.generate_graph(data, x, edeg_index)
        edge_index = self.model.modify_aug_A(aug_dense_As, ori_As, batch)

        outputs = self.model.graph_predict(data, x=x, edge_index=edge_index)
        pred = outputs['y']
        results = self._cal_loss_acc(data, pred, res=None)

        loss_all = results['loss']
        loss_all.backward() #

        self.optimizer_F.step()
        loss = results['loss']
        
        results['loss_pred'] = loss.item()
        results['loss'] = loss.item()
        results['pred'] = pred
        # results['loss_re'] = loss_re.item()

        return results


    def _train(self, train_loader, optimizer=None, clipping=None, epoch=None):
        
        loss_all = 0
        acc_all = 0

        y_true = []
        y_pred = []
        
        all_results = {}
        
        for i, data in enumerate(train_loader):

            if 'g_x' not in data and (data.x.shape[0] == 1 or data.batch[-1] == 0):
                print('x shape:', data.x.shape)
                continue
                
            if i == len(train_loader):
                print('len train_loader:', len(train_loader))
                continue
            
            data = data.to(self.device)
            ori_As = convert_to_dense_adj_custom(data.edge_index, data.batch)
            print('????????????????')
            # train graph generator:
            for i in range(self.args.inner_processes_G):
                generator_results = self.train_adj(data, ori_As)
                res_str = ''
                for k, v in generator_results.items():
                    res_str += f'train_adj {k}: {v} '
                print(res_str)

            for k, v in generator_results.items():
                all_results[k] = v

            # train predictor:
            for i in range(self.args.inner_processes_F):
                preddictor_results = self.train_predictor(data, ori_As)
                res_str = ''
                for k, v in preddictor_results.items():
                    if 'loss' in k:
                        res_str += f'train_gcn {k}: {v} '
                print(res_str)

            for k, v in preddictor_results.items():
                all_results[k] = v

            try:
                num_graphs = data.num_graphs
            except TypeError:
                num_graphs = data.adj.size(0)


            loss_all += all_results['loss_pred'] * num_graphs
            acc_all += all_results['acc'] * num_graphs


            # if self.evaluator is not None or self.roc_auc:
            pred = preddictor_results['pred']

            y_true.append(data.y.detach().cpu())
            y_pred.append(pred.detach().cpu())
            
        return self._cal_evl(train_loader, y_true, y_pred, acc_all, loss_all), all_results

    def log_grad_norm(self):
        # Calculate the gradient norm
        grad_norm = 0.0
        for param in self.model.parameters():
            if param is None or param.grad is None:
                continue
            grad_norm += torch.norm(param.grad)**2
        grad_norm = grad_norm.sqrt().item()

        # print("Total Gradient Norm:", grad_norm)
        
        def print_grad_norm(cur_module, name):
            cur_grad_norm = 0.0
            for param in cur_module.parameters():
                cur_grad_norm += torch.norm(param.grad)**2
            cur_grad_norm = cur_grad_norm.sqrt().item()
            print(f"{name} Gradient Norm:", cur_grad_norm)

        # print_grad_norm(self.model.vgae.base_gcn.first_h, 'first_h')
        # print_grad_norm(self.model.vgae.base_gcn.linears, 'linears')
        # print_grad_norm(self.model.vgae.base_gcn.convs, 'convs')
        # print_grad_norm(self.model.vgae.base_gcn.nns, 'nns')
        
        # print_grad_norm(self.model.vgae.gcn_mean, 'gcn_mean')
        
        # print_grad_norm(self.model.vgae.gcn_logstddev, 'gcn_logstddev')
        
