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
dat='NCI1'
dat='ENZYMES'


# conf_file='config_Adapter.yml'

# degree + attributes:

# dats='NCI1 ENZYMES'

dats='PATTERN'
dats='ogbg_molhiv'
dats='COLLAB REDDIT-BINARY'
dats='REDDIT-BINARY'
dats='AIDS'

model_set='GIN_lzd_attr GIN_lzd_mix GIN_lzd_degree Baseline_lzd_mlp'

dt=0530
gpu=01
model_set='GIN_lzd_degree'
model_set='GIN_lzd_attr'


dats='ogbg-molbbbp'
dats='ogbg_moltox21'
dats='MUTAG NCI1 PROTEINS DD'
dats='DD'F
dats='REDDIT-BINARY IMDB-BINARY'

dats='AIDS DD MUTAG NCI1 PROTEINS'

dats='ENZYMES'

model_set='GCN_lzd_degree'
dats='ogbg_molhiv ogbg-molbace'

dats='CIFAR10 MNIST'

model_set='GCN_lzd_attr'

for ms in ${model_set};do

conf_file=config_${ms}.yml

for dat in ${dats};do

tag=${ms}_${dat}


nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} --result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done

done