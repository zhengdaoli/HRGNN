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
dats='AIDS'
dats='ENZYMES'

dats='PROTEINS'

dats='ogbg-molbace'
# ---------
dats='ogbg_moltox21'
model_set='GIN_lzd_attr_tox21'
# ---------

dats='DD'
dats='MUTAG'

model_set='GIN_lzd_attr'

ratios='0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95'


op='add'
op='drop'
op='rewire'

model_set='DGGN_lzd_attr'


dats='NCI1'



dats='MUTAG'

dats='MUTAG PROTEINS ENZYMES NCI1'
dats='MUTAG PROTEINS ENZYMES NCI1'

dats='AIDS DD'



ratios='0.01'
ratios='0.10'

ratios='0.01 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36'

ratios='0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50'

ratios='0.70 0.80 0.90'

ratios='0.40 0.50 0.60 0.70 0.80 0.90'


ratios='0.01 0.10 0.30'
ratios='0.01'

dt=0116
model_set='GraphDGGN_lzd_attr'

patience_epochs=1

inner_processes_G=2
inner_processes_F=1

ori_ratio=0.5
aug_ratio=0.5

use_hvo="True"
k_components=4
seed=42

k_cross=1

lrG=0.002
lrP=0.01

weight_decay=5e-4
# add these parameters: self.args.l1_reg, self.args.l2_reg, self.args.feature_reg
l1_reg=0.01
l2_reg=0.01
feature_reg=0.01

proj_home=/li_zhengdao/github/DGGN_results
save_fig_path=${proj_home}/figs/${tag}_tsne.png
epochs=1
device=cuda:0
lr=0.01
gen_type=graph_hgg

conv_type='GIN'
conv_type='GCN'

ggn_gnn_type='gcn'
ggn_gnn_type='gin'
outer_folds=5

for rati in ${ratios};do
for ms in ${model_set};do
conf_file=config_${ms}.yml
for dat in ${dats};do
echo 'running '${conf_file}

# tag=${ms}_${dat}_${rati}
tag=${ms}_${dat}_${op}_${rati}_hgg_${gen_type}_innerpG${inner_processes_G}_${perturb_type}_${use_hvo}_${ori_ratio}_lrG${lrG}_${gen_type}
# pretrain_folder='result_0714_'${ms}_${dat}
# --outer-folds 1 \
# --inner-folds 1 \
# --ogb_evl True \
# --mol_split True \

# result_0718_DGGN_lzd_attr_MUTAG
log_path=${proj_home}/graph_level/logs/${dt}_${tag}_nohup.log

echo '    check log:'
# echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'
echo 'tail -f '${log_path}


# --pretrain True \
python -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--perturb_op ${op} \
--rewire_ratio ${rati} \
--dataset-name ${dat} \
--l1_reg ${l1_reg} \
--l2_reg ${l2_reg} \
--ggn_gnn_type ${ggn_gnn_type} \
--conv_type ${conv_type} \
--feature_reg ${feature_reg} \
--k_components ${k_components} \
--outer_folds ${outer_folds} \
--inner_processes_G ${inner_processes_G} \
--inner_processes_F ${inner_processes_F} \
--ori_ratio ${ori_ratio} \
--aug_ratio ${aug_ratio} \
--use_hvo ${use_hvo} \
--k_cross ${k_cross} \
--lrG ${lrG} \
--lrP ${lrP} \
--weight_decay ${weight_decay} \
--device ${device} \
--patience_epochs ${patience_epochs} \
--gen_type ${gen_type} \
--model_checkpoint_path ${proj_home}/rewiring_models/ \
--result-folder ${proj_home}/results/result_${dt}_${tag} --debug > ${log_path} 2>&1
# --result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &


done
done
done