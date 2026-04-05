#!/bin/bash

set -e

export PYTHONPATH="./:$PYTHONPATH"
export PYTHONWARNINGS="ignore::UserWarning"

dataset=$1
split=${2:-"test"}
num_samples=${3:-100}  # 注意：并行运行时，总样本数约为 num_samples * 4

# 构建参数
if [ "$num_samples" -eq 0 ]; then
    SAMPLE_ARG=""  # 不限制样本数
else
    SAMPLE_ARG="--num_samples $num_samples"
fi

model_gnd_path="model_zoo/VideoMind-2B"
model_ver_path="model_zoo/VideoMind-2B"

# 定义使用的 GPU ID
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

echo "============================================================"
echo "🧪 Ablation Study: $dataset ($split)"
echo "📊 GPUs: $NUM_GPUS (Parallel Execution)"
echo "============================================================"

# 创建结果汇总目录
result_dir="outputs_2b/ablation_${dataset}_${split}_n${num_samples}"
mkdir -p $result_dir

# ============================================================
# Ablation 1: Effect of Number of Scales
# ============================================================
echo ""
echo "============================================================"
echo "📊 Ablation 1: Effect of Number of Scales"
echo "============================================================"

for num_scales in 1 2 3; do
    echo ""
    echo "--- Running: num_scales=$num_scales ---"
    
    pred_path="${result_dir}/scales_${num_scales}"
    
    for idx in "${!GPUS[@]}"; do
        gpu=${GPUS[$idx]}
        CUDA_VISIBLE_DEVICES=$gpu python videomind/eval/infer_auto_ablation.py \
            --dataset $dataset \
            --split $split \
            --pred_path $pred_path \
            --model_gnd_path $model_gnd_path \
            --model_ver_path $model_ver_path \
            $SAMPLE_ARG \
            --num_scales $num_scales \
            --iou_threshold 0.9 \
            --verifier_topk 20 \
            --use_verifier 1 \
            --chunk $NUM_GPUS \
            --index $idx &
    done
    wait # 等待所有 GPU 任务完成
    
    echo "✅ Completed: num_scales=$num_scales"
done

# ============================================================
# Ablation 2: Effect of IoU Threshold for Deduplication
# ============================================================
echo ""
echo "============================================================"
echo "📊 Ablation 2: Effect of IoU Threshold for Deduplication"
echo "============================================================"

for iou_thr in 0.7 0.8 0.9 1.0; do
    echo ""
    echo "--- Running: iou_threshold=$iou_thr ---"
    
    pred_path="${result_dir}/iou_${iou_thr}"
    
    for idx in "${!GPUS[@]}"; do
        gpu=${GPUS[$idx]}
        CUDA_VISIBLE_DEVICES=$gpu python videomind/eval/infer_auto_ablation.py \
            --dataset $dataset \
            --split $split \
            --pred_path $pred_path \
            --model_gnd_path $model_gnd_path \
            --model_ver_path $model_ver_path \
            $SAMPLE_ARG \
            --num_scales 3 \
            --iou_threshold $iou_thr \
            --verifier_topk 20 \
            --use_verifier 1 \
            --chunk $NUM_GPUS \
            --index $idx &
    done
    wait
    
    echo "✅ Completed: iou_threshold=$iou_thr"
done

# ============================================================
# Ablation 3: Effect of Verifier Scope
# ============================================================
echo ""
echo "============================================================"
echo "📊 Ablation 3: Effect of Verifier Scope"
echo "============================================================"

# 无 Verifier
echo ""
echo "--- Running: No Verifier ---"
pred_path="${result_dir}/verifier_none"

for idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$idx]}
    CUDA_VISIBLE_DEVICES=$gpu python videomind/eval/infer_auto_ablation.py \
        --dataset $dataset \
        --split $split \
        --pred_path $pred_path \
        --model_gnd_path $model_gnd_path \
        --model_ver_path $model_ver_path \
        $SAMPLE_ARG \
        --num_scales 3 \
        --iou_threshold 0.9 \
        --verifier_topk 5 \
        --use_verifier 0 \
        --chunk $NUM_GPUS \
        --index $idx &
done
wait

echo "✅ Completed: No Verifier"

# 不同 Verifier Top-K
for topk in 5 10 20; do
    echo ""
    echo "--- Running: verifier_topk=$topk ---"
    
    pred_path="${result_dir}/verifier_top${topk}"
    
    for idx in "${!GPUS[@]}"; do
        gpu=${GPUS[$idx]}
        CUDA_VISIBLE_DEVICES=$gpu python videomind/eval/infer_auto_ablation.py \
            --dataset $dataset \
            --split $split \
            --pred_path $pred_path \
            --model_gnd_path $model_gnd_path \
            --model_ver_path $model_ver_path \
            $SAMPLE_ARG \
            --num_scales 3 \
            --iou_threshold 0.9 \
            --verifier_topk $topk \
            --use_verifier 1 \
            --chunk $NUM_GPUS \
            --index $idx &
    done
    wait
    
    echo "✅ Completed: verifier_topk=$topk"
done

# ============================================================
# 分析结果
# ============================================================
echo ""
echo "============================================================"
echo "📈 Analyzing Results..."
echo "============================================================"

python scripts/analysis/ablation_analysis.py \
    --result_dir $result_dir \
    --dataset $dataset \
    --output_file "${result_dir}/ablation_summary.md"

echo ""
echo "============================================================"
echo "🎉 Ablation Study Completed!"
echo "📁 Results saved to: $result_dir"
echo "📄 Summary: ${result_dir}/ablation_summary.md"
echo "============================================================"