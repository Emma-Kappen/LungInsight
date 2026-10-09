#!/usr/bin/env bash
set -euo pipefail

cd /home/LungInsight_v2

for seed in 123 456 789 1337 2025; do
    for attribute in all subtlety sphericity margin lobulation spiculation texture; do
        echo "===== seed=${seed} attribute=${attribute} ====="

        python backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/train_objective.py \
            --h5 datasets/LIDC/lidc_patches.h5 \
            --split-file backend/models/nodule_ranker_3d/checkpoints_model2_final/split.json \
            --out-dir backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/checkpoints/mt_vs_st_median_w0.5 \
            --objective ordinal_reader \
            --ranking-target median \
            --rank-loss-weight 0.5 \
            --ranking-pair-threshold 0.5 \
            --attribute "${attribute}" \
            --device auto \
            --batch-size 8 \
            --num-workers 4 \
            --epochs 75 \
            --patience 10 \
            --seed "${seed}"
    done
done

echo "===== SWEEP COMPLETE ====="
