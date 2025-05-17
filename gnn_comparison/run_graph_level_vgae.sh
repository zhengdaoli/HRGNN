#!/bin/bash

# ========================== 实验配置 ==========================
# 数据集列表
dats=(
    "MUTAG"
    # "NCI1"
    # "PROTEINS"
    # "ENZYMES"
    # "ogbg_molhiv"
)

# 模型集合
model_set=(
    # "GIN_lzd_attr"
    # "GraphDGGN_lzd_attr"
    # "GCN_lzd_attr"
    "DGGN_lzd_attr"
)

# 扰动操作类型
op_types=(
    # "add"
    # "drop"
    "rewire"
)

# 扰动比例列表
ratios=(
    "0.10" "0.20" "0.30" "0.50" 
    # "0.40" "0.50" "0.60"
    # "0.70" "0.80" "0.90"
)

# ========================== 固定参数 ==========================
proj_home="/root/zhengdao/github/DGGN/DGGN_results"
dt=$(date +%m%d)  # 获取当前日期（月日）
device="cuda:0"
epochs=200
lr=0.01
weight_decay=5e-4
patience_epochs=50
outer_folds=5
conv_type="GIN"  # 可选项: GIN/GCN


l1_reg=0.01
l2_reg=0.01
feature_reg=0.01

ggn_gnn_type='gin'
k_components=4

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

# gen_type=graph_hgg
gen_type=vgae

job_type_name='vgae_only'
# ========================== 实验循环 ==========================
for dat in "${dats[@]}"; do
    for model in "${model_set[@]}"; do
        conf_file="config_${model}.yml"
        
        for op in "${op_types[@]}"; do
            for ratio in "${ratios[@]}"; do
                # 生成唯一实验标识
                # tag="${model}_${dat}_${op}_${ratio}"
                tag="${model}_${dat}_${op}_${ratio}_hgg_${gen_type}_${use_hvo}_${ori_ratio}_${job_type_name}"

                log_path="${proj_home}/graph_level/logs/${tag}_nohup.log"
                result_dir="${proj_home}/results/result_${tag}"
                
                # 执行训练命令
                # --ogb_evl False \
                # --pretrain True \
                python -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
                --perturb_op ${op} \
                --rewire_ratio ${ratio} \
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
                --result-folder "${result_dir}" # \
                > "${log_path}" 2>&1
                echo "${tag} started, log is: tail -f ${log_path}"
            done
        done
    done
done

# /root/zhengdao/github/DGGN/DGGN_results/results/result_0505_GIN_lzd_attr_MUTAG_rewire_0.20/GIN_MUTAG_assessment/5_NESTED_CV/OUTER_FOLD_1/HOLDOUT_MS/config_1