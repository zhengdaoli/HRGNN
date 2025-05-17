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
dats='MUTAG NCI1 PROTEINS DD COLLAB REDDIT-BINARY'
dats='REDDIT-MULTI'

dt=0511
gpu=01

dats='CIFAR10 MNIST'

# this is multilabel, cannot split.
# dats='ogbg_moltox21' 

dats='ogbg_molhiv'

model_set='Baseline_lzd_mlp'

for ms in ${model_set};do

conf_file=config_${ms}.yml

for dat in ${dats};do

echo 'running '${conf_file}

tag=${ms}_${dat}

# --outer-folds 1 \
# --inner-folds 1 \
# --ogb_evl True \
# --mol_split True \

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} \
--result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'


done

done