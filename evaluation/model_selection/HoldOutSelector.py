import os
import json
import concurrent.futures

import logging
# from log import Logger

local_logger = logging.getLogger(__name__)

class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        print('Logger filepath:', filepath)
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock
        print('Logger mode:', self.mode)

    def log(self, str):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()


class HoldOutSelector:
    """
    Class implementing a sufficiently general framework to do model selection
    """
    def __init__(self, max_processes):
        self.max_processes = max_processes
        
        # Create the experiments folder straight away
        self._CONFIG_BASE = 'config_'
        self._CONFIG_FILENAME = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'

    def process_results(self, HOLDOUT_MS_FOLDER, no_configurations):
        best_vl = 0.
        best_i = 0
        best_config = None
        for i in range(1, no_configurations+1):
            try:
                config_filename = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(i),
                                               self._CONFIG_FILENAME)
                
                print('config filename:', config_filename)

                with open(config_filename, 'r') as fp:
                    config_dict = json.load(fp)

                vl = config_dict['VL_score']

                if best_config is None:
                    best_config = config_dict

                if best_vl <= vl:
                    best_i = i
                    best_vl = vl
                    best_config = config_dict

            except Exception as e:
                print(e)

        print('Model selection winner for experiment', HOLDOUT_MS_FOLDER, 'is config ', best_i, ':')
        for k in best_config.keys():
            print('\t', k, ':', best_config[k])

        return best_config


    def best_config_save_path(self, exp_path):
        HOLDOUT_MS_FOLDER = os.path.join(exp_path, 'HOLDOUT_MS')
        best_path = os.path.join(HOLDOUT_MS_FOLDER, self.WINNER_CONFIG_FILENAME)
        return best_path
    
    
    def load_pretrain_best_config(self, exp_path, pretrain_model_folder=None):
        # load bestconfig from file:
        # replace date:
        config_path = self.best_config_save_path(exp_path=exp_path)
        
        if pretrain_model_folder is not None:
            # 'results/result_0712_GIN_lzd_attr_NCI1_0.18/GIN_NCI1_assessment/' replace second result_0712 with pretrain_model_folder:
            config_path = config_path.split('/')
            config_path[1] = pretrain_model_folder
            config_path = '/'.join(config_path)
        
        print('load best config from path:', config_path)
        with open(config_path, 'r') as fp:
            return json.load(fp)

    
    
    def model_selection(self, dataset_getter, experiment_class, exp_path, model_configs, 
                        debug=False, other=None, outer_k=-1):
        """
        :param experiment_class: the kind of experiment used
        :param debug:
        :return: the best performing configuration on average over the k folds. TL;DR RETURNS A MODEL, NOT AN ESTIMATE!
        """
        HOLDOUT_MS_FOLDER = os.path.join(exp_path, 'HOLDOUT_MS')

        if not os.path.exists(HOLDOUT_MS_FOLDER):
            os.makedirs(HOLDOUT_MS_FOLDER)

        config_id = 0
    

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_processes)

        for config in model_configs:  # generate_grid(model_configs):

            # Create a separate folder for each experiment
            exp_config_name = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(config_id + 1))
            if not os.path.exists(exp_config_name):
                os.makedirs(exp_config_name)

            json_config = os.path.join(exp_config_name, self._CONFIG_FILENAME)
            if not os.path.exists(json_config):
                if not debug:
                    local_logger.warning(f"Starting config {config_id + 1} of {len(model_configs)}")
                    # pool.submit(self._model_selection_helper, dataset_getter, experiment_class, config,
                                # exp_config_name, other, outer_k=outer_k)
                    self._model_selection_helper(dataset_getter, experiment_class, config,
                                exp_config_name, other, outer_k=outer_k)
                else:  # DEBUG
                    self._model_selection_helper(dataset_getter, experiment_class, config, exp_config_name,
                                            other, outer_k=outer_k)
                    local_logger.warning(f"Starting _model_selection_helper, {config_id + 1} of {len(model_configs)}")
            else:
                # Do not recompute experiments for this fold.
                print(f"Config {json_config} already present! Shutting down to prevent loss of previous experiments")
                continue

            config_id += 1

        pool.shutdown()  # wait the batch of configs to terminate

        best_config = self.process_results(HOLDOUT_MS_FOLDER, config_id)

        with open(self.best_config_save_path(exp_path=exp_path), 'w') as fp:
            print('dump best_config: ', best_config)
            json.dump(best_config, fp)
            
        # with open(os.path.join(HOLDOUT_MS_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
        #     json.dump(best_config, fp)

        return best_config

    def _model_selection_helper(self, dataset_getter, experiment_class, config, exp_config_name,
                                other=None, outer_k=-1):
        """
        :param dataset_getter:
        :param experiment_class:
        :param config:
        :param exp_config_name:
        :param other:
        :return:
        """
        # Create the experiment object which will be responsible for running a specific experiment
        # local_logger.warning('Starting _model_selection_helper')
        # local_logger.warning(f"config: {config}, exp_config_name: {exp_config_name}")
        experiment = experiment_class(config, exp_config_name)
        
        # Set up a log file for this experiment (run in a separate process)
        log_path = str(os.path.join(experiment.exp_path, 'experiment.log'))
        logger = Logger(log_path, mode='a')
        logger.log('Configuration: ' + str(experiment.model_config))

        config_filename = os.path.join(experiment.exp_path, self._CONFIG_FILENAME)

        # ------------- PREPARE DICTIONARY TO STORE RESULTS -------------- #

        selection_dict = {
            'config': experiment.model_config.config_dict,
            'TR_score': 0.,
            'VL_score': 0.,
        }
        dataset_getter.set_inner_k(None)  # need to stay this way
        # local_logger.warning(experiment)

        metrics = experiment.run_valid(dataset_getter, logger, other, outer_k=outer_k)

        selection_dict['TR_score'] = float(metrics.train_acc) if metrics.train_acc is not None else -1
        selection_dict['VL_score'] = float(metrics.val_acc) if metrics.val_acc is not None else -1
        
        selection_dict['TR_roc_auc'] = float(metrics.train_roc_auc) if metrics.train_roc_auc is not None else -1
        selection_dict['VL_roc_auc'] = float(metrics.val_roc_auc) if metrics.val_roc_auc is not None else -1

        logger.log('TR Accuracy: ' + str(metrics.train_acc) + ' VL Accuracy: ' + str(metrics.val_acc)+ 
                   ' TR_roc_auc: '+ str(selection_dict['TR_roc_auc']) + ' VL_roc_auc: ' + str(selection_dict['VL_roc_auc']))

        with open(config_filename, 'w') as fp:
            json.dump(selection_dict, fp)
