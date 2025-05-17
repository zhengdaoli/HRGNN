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
dats='MUTAG NCI1 PROTEINS DD'
dats='ogbg_molhiv'
dats='CIFAR10'
dats='COLLAB REDDIT-BINARY'
dats='REDDIT-BINARY'
dats='AIDS'


model_set='GIN_lzd_attr GIN_lzd_mix GIN_lzd_degree Baseline_lzd_mlp EGNN_lzd_mix'

dt=0605
gpu=01
dats='ogbg-molbbbp'
dats='ogbg_moltox21'


dats='ogbg_moltox21 ogbg-molbace ogbg_molhiv'

model_set='EGNN_lzd_attr EGNN_lzd_mix'


paras='0.1 0.2 0.3 0.4 0.5'


dats='syn_degree'

dats='syn_cc'
model_set='Baseline_lzd_mlp_degree'

paras='0.1'
paras='0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9'

paras='0.2 0.3 0.5 0.6 0.7 0.8'


class_num='class10'
class_num='class2'

class_num='class2_final'

for ms in ${model_set};do

conf_file=config_${ms}.yml

for dat in ${dats};do

for para in ${paras};do
echo 'running '${conf_file}

tag=${ms}_${dat}_${para}_${class_num}

# --outer-folds 1 \
# --inner-folds 1 \
# --ogb_evl True \
# --mol_split True \

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} \
--dataset_para ${para}_${class_num} \
--result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done

done

done