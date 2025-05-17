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

dt=0318
gpu=00
conf_file='config_GIN_lzd_mix_adapter00.yml'
dats='IMDB-MULTI COLLAB'
dats='CIFAR10'

for dat in ${dats};do

echo 'running degree and attr decoupled: '${dat}
tag=decouple_degree_attr_${dat}

nohup python3 -u Launch_Experiments.py \
--outer-folds 1 \
--outer-processes 1 \
--inner-folds 1 \
--config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} --result-folder results/result_GIN_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo 'config file: '${conf_file}
echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done