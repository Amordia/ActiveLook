#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="/home/amax/VideoMind:$PYTHONPATH"

# 创建日志目录
mkdir -p logs

dataset=$1
split=${2:-"test"}
out_name=${3:-"${dataset}_${split}"}

model_gnd_path="model_zoo/VideoMind-2B"
model_ver_path="model_zoo/VideoMind-2B"
model_pla_path="model_zoo/VideoMind-2B"

pred_path="outputs_2b/${dataset}_${split}_self_refine_multi_scale_v2"

echo -e "\e[1;36mEvaluating:\e[0m $dataset ($split)"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

echo -e "\e[1;33m[GPU 3]\e[0m Resuming chunk index 3...\e[0m"

CUDA_VISIBLE_DEVICES=3 ASCEND_RT_VISIBLE_DEVICES=3 python videomind/eval/infer_auto.py \
    --dataset mlvu \
    --split test \
    --pred_path outputs_2b/mlvu_test_self_refine_multi_scale_v2 \
    --model_gnd_path model_zoo/VideoMind-2B \
    --model_ver_path model_zoo/VideoMind-2B \
    --model_pla_path model_zoo/VideoMind-2B \
    --chunk 4 \
    --index 3 
    # 2>&1 | tee logs/gpu3_restart_$(date +%s).log

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\e[1;32m✅ GPU 3 chunk completed\e[0m"
else
    echo -e "\e[1;31m❌ GPU 3 chunk failed (exit code: $EXIT_CODE)\e[0m"
    exit $EXIT_CODE
fi

echo ""
echo -e "\e[1;36mRunning final evaluation...\e[0m"

python videomind/eval/eval_auto.py $pred_path --dataset $dataset