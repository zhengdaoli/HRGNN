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
dat="CSL"
dat='COLLAB'
dat='REDDIT-BINARY'
dat='IMDB-BINARY' # no attribute
dat="PROTEINS"
dat='MUTAG'
dat='DD'
dat='NCI1'
dat='ENZYMES'


# conf_file='config_Adapter.yml'

# degree + attributes:

# dats='NCI1 ENZYMES'

dats='PATTERN'
dats='ogbg_molhiv'
dats='CIFAR10'
dats='COLLAB REDDIT-BINARY'
dats='REDDIT-BINARY'
dats='AIDS'


model_set='GIN_lzd_attr GIN_lzd_mix GIN_lzd_degree Baseline_lzd_mlp EGNN_lzd_mix'



dt=0522
gpu=01

dats='ogbg-molbbbp'
dats='ogbg_moltox21'
dats='MUTAG NCI1 PROTEINS DD COLLAB REDDIT-BINARY'
dats='IMDB-BINARY IMDB-MULTI'



dats='ogbg-molbace ogbg_molhiv'
dats='ogbg_ppa'
model_set='Baseline_lzd_mlp_edge_attr'

for ms in ${model_set};do

conf_file=config_${ms}.yml

for dat in ${dats};do

echo 'running '${conf_file}
tag=${ms}_${dat}
# --mol_split True \
# --outer-folds 1 \
# --inner-folds 1 \

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} \
--ogb_evl True \
--result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'


done

done