# gnn_model_name=GCN
 
proj_home=./

gnn_model_name=pro_gnn

gnn_model_name=GCN
gnn_model_name=DGGN

dt=1018
data_name=PubMed
gen_type=node_vgae
patience_epochs=30
inner_processes_G=2
inner_processes_F=1

ori_ratio=0.5
aug_ratio=0.5
k_components=5
perturb_type=ori
perturb_type=random
rewire_ratio=0.5
seed=42
use_hvo=True
k_cross=5

# if perturb_type is ori then tag is without rewire_ratio, else tag is with rewire_ratio

exp_path=${proj_home}/node_task/exp/${tag}
save_fig_path=${proj_home}/figs/${tag}_tsne.png
epochs=200
device=cuda:0

# use for loop to run different ratios:
# ratios=(0.1)

# ratios=(0.4 0.5 0.6)

ratios=(0.0)
ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
weight_decay=5e-4

for rewire_ratio in ${ratios[@]}; do
    echo $rewire_ratio
    
    if [ ${perturb_type} == "ori" ]; then
        tag=node_classification_${data_name}_${gnn_model_name}_${dt}_${perturb_type}
    else
        tag=node_classification_${data_name}_${gnn_model_name}_${dt}_raito${rewire_ratio}_innerpG${inner_processes_G}_${perturb_type}_${use_hvo}
    fi

    exp_path=${proj_home}/node_task/exp/${tag}
    save_fig_path=${proj_home}/figs/${tag}_tsne.png

    echo $proj_home/node_task/logs/${tag}.log
    echo ${tag}_tsne.png

    python -u $proj_home/node_main.py \
    --dropout=0.5 \
    --device=${device} \
    --use_hvo=${use_hvo} \
    --lr=0.01 \
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
