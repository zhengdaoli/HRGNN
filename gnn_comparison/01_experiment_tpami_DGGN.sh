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
dats='ogbg-molbbbp'
dats='ogbg_moltox21'

model_set='EGNN_lzd_attr EGNN_lzd_mix'

dats='ogbg-molbace'
dats='ogbg_moltox21'
dats='ENZYMES ogbg_moltox21'
dats='ogbg_molhiv'
dats='CIFAR10 MNIST'
dats='DD AIDS'
dats='ogbg-molbace'

dats='NCI1'

dats='MUTAG'

dats='MUTAG ENZYMES DD'

proj_home=/li_zhengdao/github/DGGN_results

dt=0801

model_set='DGGN_lzd_attr'

for ms in ${model_set};do
conf_file=config_${ms}.yml
for dat in ${dats};do
echo 'running '${conf_file}
# tag=${ms}_${dat}_${rati}
# --outer-folds 1 \
# --inner-folds 1 \
# --ogb_evl True \
# --mol_split True \

rati='0.01,0.02,04'
rati='0.1'

op='drop'
op='rewire'
# tag=${ms}_${dat}_${op}_${rati}
job_type_name='vgae_only'

tag=${ms}_${dat}_${job_type_name}

# --rewire_ratio ${rati} \
# --perturb_op ${op} \

log_path=${proj_home}/graph_level/logs/${gpu}_${dt}_${tag}_nohup.log
nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--wandb True \
--job_type_name ${job_type_name} \
--dataset-name ${dat} \
--gen_type 'vgae' \
--model_checkpoint_path ${proj_home}'/graph_level/rewiring_models' \
--result-folder ${proj_home}/graph_level/results/result_${dt}_${tag} --debug > ${log_path} 2>&1 &

echo '    check log:'
echo 'tail -f '${log_path}

done
done

# done