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
dats='ogbg_molhiv'

dats='CIFAR10 MNIST'
dats='PROTEINS'

dats='MUTAG ogbg-molbace NCI1'
dats='MUTAG'
dats='NCI1'
dats='AIDS'
dats='ENZYMES'

dats='PROTEINS'

dats='ogbg-molbace'
# ---------
dats='ogbg_moltox21'
model_set='GIN_lzd_attr_tox21'
# ---------

dats='DD'

model_set='GIN_lzd_attr'

dats='MUTAG'
model_set='DGGN_lzd_attr'

ratios='0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95'


op='add'
op='rewire'

dt=0801
ratios='0.01 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36'
op='drop'
for rati in ${ratios};do
for ms in ${model_set};do
conf_file=config_${ms}.yml
for dat in ${dats};do
echo 'running '${conf_file}

# tag=${ms}_${dat}_${rati}
tag=${ms}_${dat}_${op}_${rati}_vgae_loss
# pretrain_folder='result_0714_'${ms}_${dat}
# --outer-folds 1 \
# --inner-folds 1 \
# --ogb_evl True \
# --mol_split True \

pretrain_folder='result_0714_'${ms}_${dat} # GIN
pretrain_folder='result_0722_'${ms}_${dat}_StepLR # DGGN
pretrain_folder='result_0722_'${ms}_${dat}_StepLR # DGGN
pretrain_folder='result_0724_'${ms}_${dat}_op_drop
pretrain_folder='result_0725_'${ms}_${dat}_no_iterative

pretrain_folder='result_0731_'${ms}_${dat}_vgae_loss # DGGN + vgae loss

# result_0718_DGGN_lzd_attr_MUTAG
log_path=${proj_home}/graph_level/logs/${gpu}_${dt}_${tag}_nohup.log

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--pretrain True \
--perturb_op ${op} \
--rewire_ratio ${rati} \
--dataset-name ${dat} \
--pretrain_model_folder ${pretrain_folder} \
--model_checkpoint_path '/li_zhengdao/github/GenerativeGNN/rewiring_models' \
--result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_pretrain_nohup.log 2>&1
# --result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
# echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_pretrain_nohup.log'

done
done
done