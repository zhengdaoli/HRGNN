#!/bin/bash

# ========================== 实验配置 ==========================
# 数据集列表
dats=(
    "MUTAG"
    "NCI1"
    "PROTEINS"
    # "ENZYMES"
    # "ogbg_molhiv"
)

# 模型集合
model_set=(
    "GIN_lzd_attr"
    # "GraphDGGN_lzd_attr"
    # "GCN_lzd_attr"
)

# 扰动操作类型
op_types=(
    # "add"
    # "drop"
    "rewire"
)

# 扰动比例列表
ratios=(
    # "0.10" "0.20" "0.30" 
    "0.50" 
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

# ========================== 实验循环 ==========================
for dat in "${dats[@]}"; do
    for model in "${model_set[@]}"; do
        conf_file="config_${model}.yml"
        
        for op in "${op_types[@]}"; do
            for ratio in "${ratios[@]}"; do
                # 生成唯一实验标识
                tag="${model}_${dat}_${op}_${ratio}"
                log_path="${proj_home}/graph_level/logs/${dt}_${tag}_nohup.log"
                result_dir="${proj_home}/results/result_${dt}_${tag}"
                
                # 执行训练命令
                # --ogb_evl False \
                python -u Launch_Experiments.py \
                    --config-file "gnn_comparison/${conf_file}" \
                    --perturb_op "${op}" \
                    --rewire_ratio "${ratio}" \
                    --dataset-name "${dat}" \
                    --conv_type "${conv_type}" \
                    --outer_folds ${outer_folds} \
                    --weight_decay ${weight_decay} \
                    --device "${device}" \
                    --patience_epochs ${patience_epochs} \
                    --model_checkpoint_path "${proj_home}/rewiring_models/" \
                    --result-folder "${result_dir}" # \
                    # > "${log_path}" 2>&1

                echo "${tag} started, log is: tail -f ${log_path}"
            done
        done
    done
done

# /root/zhengdao/github/DGGN/DGGN_results/results/result_0505_GIN_lzd_attr_MUTAG_rewire_0.20/GIN_MUTAG_assessment/5_NESTED_CV/OUTER_FOLD_1/HOLDOUT_MS/config_1