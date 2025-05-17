
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
dat="CSL"
dat='REDDIT-BINARY'
dat="PROTEINS"
dat='MUTAG'
dat='COLLAB'
dat='DD'
dat='ENZYMES'
dat='IMDB-BINARY'

# python PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-degree --use-random-normal --use-pagerank --use-eigen
# python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-random-normal
# python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-degree

dat='all'
dat='ENZYMES'
dat='DD'
dat="CSL"
dat='NCI1'
dat="PROTEINS"
dat='IMDB-BINARY'
dat='REDDIT-BINARY'
dat='COLLAB'
dat='all'
dat='MUTAG'

python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-degree
# python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-random-normal
# python ./PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10


# cp -r DATA/SYNTHETIC/${dat}/ DATA/


# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name ${dat} --result-folder result_1009 --debug 

# python3 Launch_Experiments.py --config-file config_GraphSAGE.yml --dataset-name ${dat} --result-folder lzd --debug

# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name ${dat} --result-folder result_1021 --debug 

