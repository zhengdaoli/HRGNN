import os
import json

import numpy as np
import concurrent.futures

from evaluation.dataset_getter import DatasetGetter
import logging

local_logger = logging.getLogger(__name__)

from evaluation.model_selection.HoldOutSelector import Logger


class KFoldAssessment:
    """
    Class implementing a sufficiently general framework to do model ASSESSMENT
    """
    def __init__(self, outer_folds, model_selector, exp_path, model_configs, outer_processes=2):
        self.outer_folds = outer_folds
        self.outer_processes = outer_processes
        self.model_selector = model_selector
        self.model_configs = model_configs  # Dictionary with key:list of possible values

        # Create the experiments folder straight away
        self.exp_path = exp_path
        self.NESTED_FOLDER = os.path.join(exp_path, str(self.outer_folds) + '_NESTED_CV')
        
        self.OUTER_FOLD_BASE = 'OUTER_FOLD_'
        self.OUTER_RESULTS_FILENAME = 'outer_results.json'
        self.ASSESSMENT_FILENAME = 'assessment_results.json'

    def process_results(self, pretrain=False):
        outer_TR_scores = []
        outer_TS_scores = []
        outer_TR_ROCAUC = []
        outer_TE_ROCAUC = []
        
        assessment_results = {}

        for i in range(1, self.outer_folds+1):
            try:
                if pretrain:
                    config_filename = os.path.join(self.NESTED_FOLDER, self.OUTER_FOLD_BASE + str(i),
                                                   "pretrain_" + self.OUTER_RESULTS_FILENAME)
                else:
                    config_filename = os.path.join(self.NESTED_FOLDER, self.OUTER_FOLD_BASE + str(i),
                                               self.OUTER_RESULTS_FILENAME)
                
                with open(config_filename, 'r') as fp:
                    outer_fold_scores = json.load(fp)

                    outer_TR_scores.append(outer_fold_scores['OUTER_TR'])
                    outer_TS_scores.append(outer_fold_scores['OUTER_TS'])
                    outer_TR_ROCAUC.append(outer_fold_scores['OUTER_TR_ROCAUC'])
                    outer_TE_ROCAUC.append(outer_fold_scores['OUTER_TE_ROCAUC'])
                    
            except Exception as e:
                print('Exception: config_filename: ', config_filename, 'pretrain:', pretrain)
                print(e)

        outer_TR_scores = np.array(outer_TR_scores)
        outer_TS_scores = np.array(outer_TS_scores)
        outer_TR_ROCAUC = np.array(outer_TR_ROCAUC)
        outer_TE_ROCAUC = np.array(outer_TE_ROCAUC)
        
        assessment_results['avg_TR_score'] = outer_TR_scores.mean()
        assessment_results['std_TR_score'] = outer_TR_scores.std()
        assessment_results['avg_TS_score'] = outer_TS_scores.mean()
        assessment_results['std_TS_score'] = outer_TS_scores.std()

        
        assessment_results['avg_TR_ROCAUC'] = outer_TR_ROCAUC.mean()
        assessment_results['std_TR_ROCAUC'] = outer_TR_ROCAUC.std()
        assessment_results['avg_TE_ROCAUC'] = outer_TE_ROCAUC.mean()
        assessment_results['std_TE_ROCAUC'] = outer_TE_ROCAUC.std()
        

        if pretrain:
            with open(os.path.join(self.NESTED_FOLDER, "pretrain_"+self.ASSESSMENT_FILENAME), 'w') as fp:
                json.dump(assessment_results, fp)
        else:
            with open(os.path.join(self.NESTED_FOLDER, self.ASSESSMENT_FILENAME), 'w') as fp:
                json.dump(assessment_results, fp)

    def risk_assessment(self, experiment_class, debug=False, other=None, repeat=False, pretrain=False):
        """
        :param experiment_class: the kind of experiment used
        :param debug:
        :param other: anything you want to share across processes
        :return: An average over the outer test folds. RETURNS AN ESTIMATE, NOT A MODEL!!!
        """
        print('pretrain ', "-----"* 10, pretrain)
        
        if not os.path.exists(self.NESTED_FOLDER):
            os.makedirs(self.NESTED_FOLDER)

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.outer_processes)
        local_logger.warning(f"Starting outer folds with {self.outer_processes} processes")

        for outer_k in range(self.outer_folds):
            
            # Create a separate folder for each experiment
            kfold_folder = os.path.join(self.NESTED_FOLDER, self.OUTER_FOLD_BASE + str(outer_k + 1))
            

            local_logger.warning(f"Creating folder {kfold_folder}")
            if not os.path.exists(kfold_folder):
                os.makedirs(kfold_folder)
            local_logger.warning(f"Folder {kfold_folder} created")
            
            if pretrain:
                json_outer_results = os.path.join(kfold_folder, "pretrain_" + self.OUTER_RESULTS_FILENAME)
            else:
                json_outer_results = os.path.join(kfold_folder, self.OUTER_RESULTS_FILENAME)
                
            local_logger.warning(f"json_outer_results: {json_outer_results}")
            
            if not os.path.exists(json_outer_results) or pretrain:
                if not debug:
                    local_logger.warning(f"Starting outer fold {outer_k + 1} of {self.outer_folds}")
                    # pool.submit(self._risk_assessment_helper, 0 if repeat else outer_k,
                    #             experiment_class, kfold_folder, debug, other, pretrain=pretrain)
                    self._risk_assessment_helper(0 if repeat else outer_k,
                                experiment_class, kfold_folder, debug, other, pretrain=pretrain)
                else:  # DEBUG
                    local_logger.warning(f"Starting _risk_assessment_helper, {outer_k + 1} of {self.outer_folds}")
                    self._risk_assessment_helper(0 if repeat else outer_k, experiment_class,
                                                kfold_folder, debug, other, pretrain=pretrain)
            else:
                # Do not recompute experiments for this outer fold.
                print(f"File {json_outer_results} already present! Shutting down to prevent loss of previous experiments")
                continue
            local_logger.warning(f"finished outer fold {outer_k + 1} of {self.outer_folds}")
            
            # Create a separate folder for each experiment
            # kfold_folder = os.path.join(self.__NESTED_FOLDER, self.__OUTER_FOLD_BASE + str(outer_k + 1))
            # if not os.path.exists(kfold_folder):
            #     os.makedirs(kfold_folder)
            # else:
            #     # Do not recompute experiments for this outer fold.
            #     print(f"Outer folder {outer_k} already present! Shutting down to prevent loss of previous experiments")
            #     continue

        pool.shutdown()  # wait the batch of configs to terminate

        self.process_results(pretrain)

    def _risk_assessment_helper(self, outer_k, experiment_class, exp_path, debug=False, other=None, pretrain=False):
        
        local_logger.warning(f"_risk_assessment_helper {outer_k + 1} of {self.outer_folds}")

        dataset_getter = DatasetGetter(outer_k)

        if pretrain:
            # load from disk:
            # local_logger.warning(f"Loading pretrain model from {self.model_configs[0]['pretrain_model_folder']}")
            best_config = self.model_selector.load_pretrain_best_config(exp_path,
                                                                        pretrain_model_folder=self.model_configs[0]['pretrain_model_folder'])
            
            # add pretrain_model_date to the best_config
            best_config['config']['pretrain_model_date'] = self.model_configs[0]['pretrain_model_date']
            best_config['config']['pretrain_model_folder'] = self.model_configs[0]['pretrain_model_folder']
            best_config['config']['rewire_ratio'] = self.model_configs[0]['rewire_ratio']
            best_config['config']['pretrain'] = self.model_configs[0]['pretrain']
            best_config['config']['perturb_op'] = self.model_configs[0]['perturb_op']
            
        else:
            # local_logger.warning(f"Starting model selection for outer fold {outer_k + 1} of {self.outer_folds}")
            best_config = self.model_selector.model_selection(dataset_getter, experiment_class, exp_path,
                                                          self.model_configs, debug, other, outer_k=outer_k)
        # local_logger.warning(f"Best config: {best_config['config']}")
        # Retrain with the best configuration and test
        experiment = experiment_class(best_config['config'], exp_path)
        
        # Set up a log file for this experiment (run in a separate process)
        if pretrain:
            logger = Logger(str(os.path.join(experiment.exp_path, 'experiment_pretrain.log')), mode='a')
        else:
            logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')
        # local_logger.warning(f"Starting experiment with best config {best_config['config']}")
        dataset_getter = DatasetGetter(outer_k)
        dataset_getter.set_inner_k(None)       # needs to stay None

        training_scores, test_scores = [], []
        training_ruc_aucs, test_ruc_aucs = [], []

        
        # Mitigate bad random initializations
        for i in range(3):
            # local_logger.warning(f"Starting experiment with best config {best_config['config']}, run {i + 1}")

            metrics = experiment.run_test(dataset_getter, logger, other, round=i, pretrain=pretrain)
            print(f"Final training run {i + 1}: {metrics.train_acc}, {metrics.test_acc}, {metrics.train_roc_auc}, {metrics.test_roc_auc},")

            training_scores.append(metrics.train_acc)
            test_scores.append(metrics.test_acc)

            training_ruc_aucs.append(metrics.train_roc_auc)
            test_ruc_aucs.append(metrics.test_roc_auc)
                               
        training_score = sum(training_scores) / len(training_scores)
        test_score = sum(test_scores) / len(test_scores)
        
        training_ruc_auc = np.mean(np.array(training_ruc_aucs)) if training_ruc_aucs[0] is not None else -1
        test_ruc_auc = np.mean(np.array(test_ruc_aucs)) if training_ruc_aucs[0] is not None else -1
        
        # TODO: add roc:
        logger.log(f"End of Outer fold. TR score:  {training_score} TS score: {test_score} \
            training_ruc_auc: {training_ruc_auc}, test_ruc_auc: {test_ruc_auc}")

        if pretrain:
            out_result_path = os.path.join(exp_path, "pretrain_" + self.OUTER_RESULTS_FILENAME)
        else:
            out_result_path = os.path.join(exp_path, self.OUTER_RESULTS_FILENAME)
            
        print('pretrain: ', pretrain, ' out_result_path: ', out_result_path)
        
        with open(out_result_path, 'w') as fp:
            json.dump({'best_config': best_config, 'OUTER_TR': training_score,
                    'OUTER_TS': test_score,
                    'OUTER_TR_ROCAUC': training_ruc_auc,
                    'OUTER_TE_ROCAUC': test_ruc_auc
                    }, fp)


