from models.gnn_wrapper.NetWrapper import NetWrapper, DGGNNetWrapper

from experiments.Experiment import Experiment
from models.utils.EarlyStopper import ClassificationMetrics
import torch
import os
import argparse

def get_sampleweight(labels):
    total = len(labels)
    class_sample_count = torch.unique(torch.tensor(labels), return_counts=True)[1]
    weight = class_sample_count.float()/total
    # need normalize.
    print('weight: ', weight, weight.shape)
    return weight

class EndToEndExperiment(Experiment):

    def __init__(self, model_configuration, exp_path):
        super(EndToEndExperiment, self).__init__(model_configuration, exp_path)
        self.exp_path = exp_path
        print('exp_path: ', exp_path)


    def init_dataset(self):
        dataset_class = self.model_config.dataset  # dataset_class()
        self.dataset_name = dataset_class.__name__
        print('dataset_class:', dataset_class)


        if 'dense' in self.model_config:
            dataset = dataset_class(dense=self.model_config.dense, config=self.model_config)
        elif 'additional_features' in self.model_config:
            print('create node additional_features:', self.model_config.additional_features)
            dataset = dataset_class(additional_features = self.model_config.additional_features, 
                                    config=self.model_config)
        elif 'additional_graph_features' in self.model_config:
            print('create additional_graph_features', self.model_config.additional_graph_features)
            dataset = dataset_class(additional_graph_features = self.model_config.additional_graph_features, config=self.model_config)
        else:
            print('model config:', self.model_config)
            dataset = dataset_class(config=self.model_config)
            
        return dataset
        
        
    def run_valid(self, dataset_getter, logger, other=None, outer_k=-1) -> ClassificationMetrics:
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """
        # print(self.model_config, dataset_getter.outer_k, dataset_getter.inner_k)

      
        model_class = self.model_config.model
        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        args = argparse.Namespace(**self.model_config.config)

        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True
        dataset = self.init_dataset()
        
        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)
        
        print('dataset dim features', dataset.fea_dim)
        print('dataset edge_attr_dim', dataset.edge_attr_dim)
        
        
        model = model_class(fea_dim=dataset.fea_dim, edge_attr_dim=dataset.edge_attr_dim, 
                            target_dim=dataset.target_dim, config=self.model_config)

        model = model.to(args.device)
        
        self.model = model
        
        # NOTE: add weight
        labels = dataset.get_labels()
        # labels = [i.y for i in train_loader.dataset]
        # labels = train_loader.data.y
        # weight = get_sampleweight(labels)
        # weight = weight.to(self.model_config['device'])
        
        weight = None
        if 'wandb' in self.model_config:
            use_wandb = self.model_config['wandb']
        # transform config dict to argparse:
        print('--------- self model config:', self.model_config)
        print('args:', args)


        if args.gen_type == 'graph_hgg':
            net = DGGNNetWrapper(model, loss_function=loss_class(weight=weight),device=self.model_config['device'],
                            config=self.model_config, exp_path=self.exp_path, outer_k=outer_k, use_wandb=use_wandb)
        else:
            net = NetWrapper(model, loss_function=loss_class(weight=weight), 
                         device=self.model_config['device'], config=self.model_config,
                         exp_path=self.exp_path, outer_k=outer_k, use_wandb=use_wandb)
        if args.gen_type != 'graph_hgg':
            only_vgae= False
            # NOTE: only update model.vgae parameters:
            if only_vgae:
                optimizer = optim_class(model.vgae.parameters(),
                                    lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])
            else:
                optimizer = optim_class(model.parameters(),
                                    lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

            if sched_class is not None:
                scheduler = sched_class(optimizer)
            else:
                scheduler = None
        else:
            optimizer = None
            scheduler = None
        
        metrics = net.train(train_loader=train_loader,
                                                    max_epochs=self.model_config['classifier_epochs'],
                                                    optimizer=optimizer, scheduler=scheduler,
                                                    clipping=clipping,
                                                    validation_loader=val_loader,
                                                    early_stopping=stopper_class,
                                                    logger=logger)
        return metrics

    def init_model(self, dataset):
        model_class = self.model_config.model
        print('--------- self model config:', self.model_config)
        model = model_class(fea_dim=dataset.fea_dim, 
                            edge_attr_dim=dataset.edge_attr_dim,
                            target_dim=dataset.target_dim,
                            config=self.model_config)
        return model


    def run_test(self, dataset_getter, logger, other=None, round=0, pretrain=False) -> ClassificationMetrics:
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR TRAINING OR EARLY STOPPING!
        :return: (training accuracy, test accuracy)
        """
        dataset = self.init_dataset()
        args = argparse.Namespace(**self.model_config.config)
            
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)
        test_loader = dataset_getter.get_test(dataset, self.model_config['batch_size'], shuffle=shuffle)

        labels = [i.y for i in train_loader.dataset]
        # weight = get_sampleweight(labels)
        # weight = weight.to(self.model_config['device'])
        weight = None

        print('--------- self model config:', self.model_config)
        model = self.init_model(dataset)
        model.to(args.device)
        
        weight = None
        if 'wandb' in self.model_config:
            use_wandb = self.model_config['wandb']
        optimizer = None
        scheduler = None
        if args.gen_type == 'graph_hgg':
            net = DGGNNetWrapper(model, loss_function=loss_class(weight=weight),device=self.model_config['device'],
                            config=self.model_config, exp_path=self.exp_path, use_wandb=use_wandb, pretrain=pretrain)
        else:
            net = NetWrapper(model, loss_function=loss_class(weight=weight), 
                         device=self.model_config['device'], config=self.model_config,
                         exp_path=self.exp_path, use_wandb=use_wandb, pretrain=pretrain)
            
        
            optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

            if sched_class is not None:
                scheduler = sched_class(optimizer)
            else:
                scheduler = None

        metrics = net.train(train_loader=train_loader, max_epochs=self.model_config['classifier_epochs'],
                      optimizer=optimizer, scheduler=scheduler, clipping=clipping,
                      validation_loader=val_loader, test_loader=test_loader, early_stopping=stopper_class,
                      logger=logger, need_save_model=True)

        return metrics
