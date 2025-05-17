# gnn_model_name=GCN
 


proj_home=./



gnn_model_name=pro_gnn
gnn_model_name=GCN
gnn_model_name=LearnGCN
gnn_model_name=DGGN

dt=1101
data_name=polblogs

data_name=CiteSeer

data_name=Cora

gen_type=node_vgae

gen_type=node_hgg

patience_epochs=100

inner_processes_G=10
inner_processes_F=1

ori_ratio=0.5
aug_ratio=0.5

use_hvo="True"

k_components=4
perturb_type=ori
perturb_type=random
seed=42

k_cross=1

lrG=0.002
lrP=0.01

weight_decay=5e-4
# add these parameters: self.args.l1_reg, self.args.l2_reg, self.args.feature_reg
l1_reg=0.01
l2_reg=0.01
feature_reg=0.01

# if perturb_type is ori then tag is without rewire_ratio, else tag is with rewire_ratio

exp_path=${proj_home}/node_task/exp/${tag}
save_fig_path=${proj_home}/figs/${tag}_tsne.png
epochs=800
device=cuda:0
lr=0.01


ratios=(0.9)
ratios=(0.5)

ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)


# epochs=100, gamma=1, hidden=16, inner_steps=2, lambda_=0, lr=0.01, lr_adj=0.01, no_cuda=False, only_gcn=False, outer_steps=1, phi=0, ptb_rate=0.3, seed=15, symmetric=False, weight_decay=0.0005)
for rewire_ratio in ${ratios[@]}; do
    echo $rewire_ratio
    
    if [ ${perturb_type} == "ori" ]; then
        tag=node_classification_${data_name}_${gnn_model_name}_${dt}_${perturb_type}
    else
        tag=node_classification_${data_name}_${gnn_model_name}_${dt}_raito${rewire_ratio}_innerpG${inner_processes_G}_${perturb_type}_${use_hvo}_${ori_ratio}_lrG${lrG}_${gen_type}
    fi

    exp_path=${proj_home}/node_task/exp/${tag}
    save_fig_path=${proj_home}/figs/${tag}_tsne.png

    echo $proj_home/node_task/logs/${tag}.log

    python -u node_main.py \
    --dropout=0.5 \
    --device=${device} \
    --use_hvo=${use_hvo} \
    --lr=${lr} \
    --lrG=${lrG} \
    --lrP=${lrP} \
    --l1_reg=${l1_reg} \
    --l2_reg=${l2_reg} \
    --feature_reg=${feature_reg} \
    --weight_decay=${weight_decay} \
    --k_cross=${k_cross} \
    --seed=${seed} \
    --epochs=${epochs} \
    --k_components=${k_components} \
    --exp_path=${exp_path} \
    --ori_ratio=${ori_ratio} \
    --aug_ratio=${aug_ratio} \
    --inner_processes_G=${inner_processes_G} \
    --inner_processes_F=${inner_processes_F} \
    --patience_epochs=${patience_epochs} \
    --best_model_save_path=${proj_home}/node_task/best_models/${tag}.pt \
    --gen_type=${gen_type} \
    --model_name=${gnn_model_name} \
    --rewire_ratio=${rewire_ratio} \
    --tsne_save_path=${save_fig_path} \
    --dataset_name=${data_name} \
    > $proj_home/node_task/logs/${tag}.log
done



# nohup python -u $proj_home/node_main.py \
# --dropout=0.4 \
# --lr=0.01 \
# --use_hvo=${use_hvo} \
# --seed=${seed} \
# --epochs=500 \
# --k_components=${k_components} \
# --exp_path=${exp_path} \
# --ori_ratio=${ori_ratio} \
# --aug_ratio=${aug_ratio} \
# --inner_processes_G=${inner_processes_G} \
# --inner_processes_F=${inner_processes_F} \
# --patience_epochs=${patience_epochs} \
# --best_model_save_path=${proj_home}/node_task/best_models/${tag}.pt \
# --gen_type=${gen_type} \
# --model_name=${gnn_model_name} \
# --rewire_ratio=${rewire_ratio} \
# --tsne_save_path=${save_fig_path} \
# --dataset_name=${data_name} \
# > $proj_home/${tag}.log 2>&1 &

# echo $proj_home/${tag}.log
# echo ${tag}_tsne.png
