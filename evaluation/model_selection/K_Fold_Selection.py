import os
import json

import numpy as np
import concurrent.futures
from copy import deepcopy

from log.Logger import Logger


class KFoldSelection:
    """
    Class implementing a sufficiently general framework to do model selection
    """

    def __init__(self, folds, max_processes):
        self.folds = folds
        self.max_processes = max_processes

        # Create the experiments folder straight away
        self._CONFIG_BASE = 'config_'
        self._CONFIG_FILENAME = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'

    def process_results(self, KFOLD_FOLDER, no_configurations):

        best_avg_vl = 0.
        best_std_vl = 100.

        for i in range(1, no_configurations+1):
            try:
                config_filename = os.path.join(KFOLD_FOLDER, self._CONFIG_BASE + str(i), self._CONFIG_FILENAME)

                with open(config_filename, 'r') as fp:
                    config_dict = json.load(fp)

                avg_vl = config_dict['avg_VL_score']
                std_vl = config_dict['std_VL_score']

                if (best_avg_vl < avg_vl) or (best_avg_vl == avg_vl and best_std_vl > std_vl):
                    best_i = i
                    best_avg_vl = avg_vl
                    best_config = config_dict

            except Exception as e:
                print(e)

        print('Model selection winner for experiment', KFOLD_FOLDER, 'is config ', best_i, ':')
        for k in best_config.keys():
            print('\t', k, ':', best_config[k])

        return best_config
    
    def best_model_save_path(self, exp_path):
        KFOLD_FOLDER = os.path.join(exp_path, str(self.folds) + '_FOLD_MS')
        best_path = os.path.join(KFOLD_FOLDER, self.WINNER_CONFIG_FILENAME)
        return best_path
            

    def model_selection(self, dataset_getter, experiment_class, exp_path, model_configs, 
                        debug=False, other=None, pretrain=False, outer_k=0):
        """
        :param experiment_class: the kind of experiment used
        :param debug:
        :return: the best performing configuration on average over the k folds. TL;DR RETURNS A MODEL, NOT AN ESTIMATE!
        """

        exp_path = exp_path
        KFOLD_FOLDER = os.path.join(exp_path, str(self.folds) + '_FOLD_MS')

        if not os.path.exists(KFOLD_FOLDER):
            os.makedirs(KFOLD_FOLDER)

        config_id = 0

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_processes)
        for config in model_configs:

            # I need to make a copy of this dictionary
            # It seems it gets shared between processes!
            cfg = deepcopy(config)

            # Create a separate folder for each experiment
            exp_config_name = os.path.join(KFOLD_FOLDER, self._CONFIG_BASE + str(config_id + 1))
            if not os.path.exists(exp_config_name):
                os.makedirs(exp_config_name)
        
            # if pretrain, then skip training
            if not debug:
                pool.submit(self._model_selection_helper, dataset_getter, experiment_class, cfg,
                            exp_config_name, other, outer_k)
            else:  # DEBUG
                self._model_selection_helper(dataset_getter, experiment_class, cfg,
                                            exp_config_name, other)

            config_id += 1

        pool.shutdown()
        
        best_config = self.process_results(KFOLD_FOLDER, config_id)

        with open(self.best_model_save_path(exp_path=exp_path), 'w') as fp:
            print('dump best_config: ', best_config)
            json.dump(best_config, fp)

        return best_config

    def _model_selection_helper(self, dataset_getter, experiment_class, config, exp_config_name,
                                other=None, outer_k=-1):

        # Set up a log file for this experiment (run in a separate process)
        logger = Logger(str(os.path.join(exp_config_name, 'experiment.log')), mode='a')

        logger.log('Configuration: ' + str(config))

        config_filename = os.path.join(exp_config_name, self._CONFIG_FILENAME)

        # ------------- PREPARE DICTIONARY TO STORE RESULTS -------------- #

        k_fold_dict = {
            'config': config,
            'folds': [{} for _ in range(self.folds)],
            'avg_TR_score': 0.,
            'avg_VL_score': 0.,
            'std_TR_score': 0.,
            'std_VL_score': 0.,
            'avg_TR_roc_auc': 0.,
            'avg_VL_roc_auc': 0.,
        }

        for k in range(self.folds):

            dataset_getter.set_inner_k(k)

            fold_exp_folder = os.path.join(exp_config_name, 'FOLD_' + str(k + 1))
            # Create the experiment object which will be responsible for running a specific experiment
            experiment = experiment_class(config, fold_exp_folder)

            metrics = experiment.run_valid(dataset_getter, logger, other, outer_k=outer_k)

            logger.log(f'{str(k+1)} split, TR Accuracy:  {metrics.train_acc} \
                        TR ROCAUC: {metrics.train_roc_auc} \
                        VL Accuracy:  {metrics.val_acc} \
                        VL ROCAUC: {metrics.val_roc_auc}')

            k_fold_dict['folds'][k]['TR_score'] = metrics.train_acc
            k_fold_dict['folds'][k]['VL_score'] = metrics.val_acc
            
            k_fold_dict['folds'][k]['TR_roc_auc'] = metrics.train_roc_auc
            k_fold_dict['folds'][k]['VL_roc_auc'] = metrics.val_roc_auc

        tr_scores = np.array([k_fold_dict['folds'][k]['TR_score'] for k in range(self.folds)])
        vl_scores = np.array([k_fold_dict['folds'][k]['VL_score'] for k in range(self.folds)])
        tr_roc_auc = np.array([k_fold_dict['folds'][k]['TR_roc_auc'] for k in range(self.folds)])
        vl_roc_auc = np.array([k_fold_dict['folds'][k]['VL_roc_auc'] for k in range(self.folds)])

        k_fold_dict['avg_TR_score'] = tr_scores.mean()
        k_fold_dict['std_TR_score'] = tr_scores.std()
        k_fold_dict['avg_VL_score'] = vl_scores.mean()
        k_fold_dict['std_VL_score'] = vl_scores.std()
        
        k_fold_dict['avg_TR_roc_auc'] = tr_roc_auc.mean()
        k_fold_dict['std_TR_roc_auc'] = tr_roc_auc.std()
        
        k_fold_dict['avg_VL_roc_auc'] = vl_roc_auc.mean()
        k_fold_dict['std_VL_roc_auc'] = vl_roc_auc.std()
        
        logger.log(f"TR avg is {k_fold_dict['avg_TR_score']} std is {k_fold_dict['std_TR_score']} \
                   VL avg is {k_fold_dict['avg_VL_score']} std is {k_fold_dict['std_VL_score']} \
                   TR ruc_auc avg is {k_fold_dict['avg_TR_roc_auc']} std is {k_fold_dict['std_TR_roc_auc']} \
                   VL ruc_auc avg is {k_fold_dict['avg_VL_roc_auc']} std is {k_fold_dict['std_VL_roc_auc']}")
        

        with open(config_filename, 'w') as fp:
            json.dump(k_fold_dict, fp)
