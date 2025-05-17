
#    CHEMICAL:
#         NCI1
#         DD
#         ENZYMES
#         PROTEINS
#    SOCIAL[_1 | _DEGREE]:
#         IMDB-BINARY
#         IMDB-MULTI
#         REDDIT-BINARY
#         REDDIT-MULTI-5K
#         COLLAB
dat='all'
dat='NCI1'
dat='ENZYMES'
dat='DD'
dat="CSL"
dat="PROTEINS"
dat='IMDB-BINARY'
dat='COLLAB'
dat='REDDIT-BINARY'
dat='MUTAG'

# python3 PrepareDatasets.py DATA/SYNTHETIC --dataset-name ${dat} --outer-k 10 --use-degree
# python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-random-normal
python3 gnn_comparison/PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10

# cp -r DATA/SYNTHETIC/${dat}/ DATA/


# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name ${dat} --result-folder result_1009 --debug 

# python3 Launch_Experiments.py --config-file config_GraphSAGE.yml --dataset-name ${dat} --result-folder lzd --debug

# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name ${dat} --result-folder result_1021 --debug 

# TODO: test all datasets using all models.

# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name all --result-folder result_1009 --debug 
# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name all --result-folder result_1009 --debug 
# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name all --result-folder result_1009 --debug 
# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name all --result-folder result_1009 --debug 




# 2022.10.09. dataset: IMDB-BINARY, GraphSAGE. (GIN better in paper)
#result_1009/GraphSAGE_IMDB-BINARY_assessment/10_NESTED_CV/OUTER_FOLD_1/HOLDOUT_MS/winner_o  {"config": {"model": "GraphSAGE", "device": "cuda:0", "batch_size": 32, "learning_rate": 0.001, "l2": 0.0, "classifier_epochs": 200, "optimizer": "Adam", "scheduler": null, "loss": "MulticlassClassificationLoss", "gradient_clipping": null, "early_stopper": {"class": "Patience", "args": {"patience": 100, "use_loss": false}}, "shuffle": true, "dim_embedding": 32, "num_layers": 3, "aggregation": "sum", "additional_features": "degree,tri_cycle", "dataset": "IMDB-BINARY"}, "TR_score": 66.04938271604938, "VL_score": 82.22222290039062}# 






# cat result_1009/pre_results/GraphSAGE_IMDB-BINARY_assessment/10_NESTED_CV/OUTER_FOLD_*/outer_results.json 
# cat result_1009/GraphSAGE_IMDB-BINARY_assessment/10_NESTED_CV/OUTER_FOLD_*/outer_results.json 


# 2022.10.20, test graph-level features.
# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_Baseline_lzd_mlp.yml --dataset-name ${dat} --result-folder result_1022 --debug 

# 2022.10.21, test graph-level features test on other datasets to prove GNN is not better than MLP if degree information is enough for the tasks.
# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_Baseline_lzd_mlp.yml --dataset-name ${dat} --result-folder result_1021 --debug 


# 2022.10.24,
# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/01_config_Baseline_lzd_mlp.yml --dataset-name ${dat} --result-folder result_1024 --debug 



