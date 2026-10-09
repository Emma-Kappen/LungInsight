#!/usr/bin/env bash
set -euo pipefail

cd /home/LungInsight_v2

for group in lobulation_spiculation subtlety sphericity margin texture; do
    echo "============================================================"
    echo "TRAINING GROUP: ${group}"
    echo "============================================================"

    python backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/train_objective.py \
        --h5 datasets/LIDC/lidc_patches.h5 \
        --split-file backend/models/nodule_ranker_3d/checkpoints_model2_final/split.json \
        --out-dir backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/checkpoints \
        --objective ordinal_reader \
        --attribute-group "${group}" \
        --ranking-target median \
        --rank-loss-weight 0.5 \
        --ranking-pair-threshold 0.5 \
        --device cuda \
        --batch-size 8 \
        --num-workers 4 \
        --epochs 75 \
        --patience 10 \
        --seed 123

    echo "COMPLETED: ${group}"
done

echo "============================================================"
echo "ALL FIVE HYBRID MODELS COMPLETED"
echo "============================================================"
