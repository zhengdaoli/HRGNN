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
dat='DD'



# conf_file='config_Adapter.yml'

# degree + attributes:

# dats='NCI1 ENZYMES'

dats='PATTERN'
dats='ogbg_molhiv'
dats='CIFAR10'
dats='COLLAB REDDIT-BINARY'
dats='REDDIT-BINARY'

model_set='GIN_lzd_attr GIN_lzd_mix GIN_lzd_degree Baseline_lzd_mlp EGNN_lzd_mix'

gpu=01
dats='ogbg_moltox21'

model_set='EGNN_lzd_attr EGNN_lzd_mix'

model_set='GCN_lzd_degree'
model_set='GCN_lzd_allone'
model_set='GCN_lzd_degree'
dats='ogbg-molbace'
dats='ogbg_moltox21'
dats='CIFAR10 MNIST'
dats='PROTEINS'
dats='MUTAG ogbg-molbace NCI1'
dats='MUTAG NCI1'
dats='NCI1'
dats='DD AIDS'
dats='MUTAG'
dats='NCI1'
dats='ogbg-molbace PROTEINS'
dats='ENZYMES'


dt=0715
dats='ogbg_molhiv'
model_set='GIN_lzd_attr'

for ms in ${model_set};do
conf_file=config_${ms}.yml
for dat in ${dats};do
echo 'running '${conf_file}

tag=${ms}_${dat}

# tag=${ms}_${dat}_${rati}
# --outer-folds 1 \
# --inner-folds 1 \
# --ogb_evl True \
# --mol_split True \

pretrain_folder='result_0714_'${ms}_${dat}

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} \
--model_checkpoint_path '/li_zhengdao/github/GenerativeGNN/rewiring_models' \
--result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done
done