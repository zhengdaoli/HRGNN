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
dats='COLLAB REDDIT-BINARY'
dats='REDDIT-BINARY'

model_set='GIN_lzd_attr GIN_lzd_mix GIN_lzd_degree Baseline_lzd_mlp EGNN_lzd_mix'

gpu=01
dats='ogbg-molbbbp'
dats='ogbg_moltox21'


model_set='GCN_lzd_degree'
model_set='GCN_lzd_allone'
model_set='GCN_lzd_degree'
model_set='EGNN_lzd_attr EGNN_lzd_mix'
dats='ogbg_molhiv'
dats='MUTAG NCI1 DD AIDS'

dats='MUTAG ogbg-molbace NCI1'
model_set='GIN_lzd_attr'
dats='ogbg-molbace'


dats='ogbg_moltox21'
model_set='GIN_lzd_attr_tox21'


dt=0714

dats='ogbg_ppa'
model_set='GIN_lzd_attr_edge'
# for rati in ${ratios};do

for ms in ${model_set};do
conf_file=config_${ms}.yml
for dat in ${dats};do
echo 'running '${conf_file}

tag=${ms}_${dat}
# tag=${ms}_${dat}_${rati}
# except for tox21:
# --repeat True \
# --inner-folds 1 \
# --rewire_ratio ${rati} \

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--mol_split True \
--outer-folds 1 \
--ogb_evl True \
--dataset-name ${dat} \
--model_checkpoint_path '/li_zhengdao/github/GenerativeGNN/rewiring_models' \
--result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done
done
# done