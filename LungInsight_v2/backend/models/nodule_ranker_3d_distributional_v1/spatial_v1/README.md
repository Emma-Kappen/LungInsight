# LungInsight v2 — spatial_objective_v1

Controlled objective ablation using the fixed LIDC H5, canonical patient-disjoint split, and the stride-8 spatial-attention architecture.

## Objective A
`ordinal_reader`

Six ordered attributes use cumulative-link ordinal supervision from all valid individual reader ratings. `internal_structure` (K=4) and `calcification` (K=6) use empirical reader distributions with soft categorical cross-entropy.

## Objective B
`mean_regression`

Six ordered attributes regress directly to the per-nodule valid-reader mean. The same nominal heads/loss remain unchanged.

Both objectives add confidence-weighted pairwise logistic ranking loss by default (`--rank-loss-weight 0.5`). Select the ranking target with `--ranking-target mean|median|mean_median_blend`; weak pairs are excluded using `--ranking-pair-threshold` (default `0.5`). Training prints TRAIN-only pair counts at thresholds 0.25, 0.50, and 1.00. Validation still reports ranking metrics against the per-nodule reader mean.

## Reader-derived probabilistic pairwise ranking

Add `--ranking-target reader_pairwise` to train a new ranking experiment without changing the existing targets or model architecture. For each attribute and nodule pair, all valid reader-rating cross-comparisons are counted, ties are ignored, and the soft target is `p_ij = wins_i / (wins_i + wins_j)`. Pairs with no non-tied comparisons are invalid. Confidence is `2 * abs(p_ij - 0.5)`; pairs below `--ranking-min-confidence` (default `0.10`) are excluded. Pair loss is confidence-weighted `BCEWithLogits(s_i - s_j, p_ij)`, where each attribute score is the model's expected rating. This also applies to the two nominal attributes' expected ratings.

Ordinal-reader supervision remains active for the six ordered attributes, and nominal reader-distribution supervision remains active for the two nominal attributes. The ranking term is added with `--rank-loss-weight`; the legacy `--ranking-pair-threshold` setting is ignored by `reader_pairwise` and remains unchanged for the original three modes.

Every epoch records full-split TRAIN and validation `reader_pairwise_loss`, `average_pair_confidence`, `valid_pair_count`, and `pair_accuracy`. Pair accuracy uses `delta > 0` against the hard target `p_ij > 0.5`. TRAIN diagnostics use a post-epoch inference pass so their scores come from one consistent model state. Checkpoints include both splits' pair diagnostics and the validation values at top level. The held-out TEST split remains unopened during training.

Training loss compares pairs within each minibatch, costing `O(A * B^2 * R^2)` comparisons for `A` attributes, batch size `B`, and `R` reader slots. Full-split diagnostics use bounded-memory blocks with rating histograms: `O(A * K * N^2)` time and `O(block_size * N)` temporary memory, where `K` is the maximum rating cardinality and `N` is the split size. The additional TRAIN inference pass and pair diagnostics increase per-epoch runtime; the CNN architecture and split are unchanged.

Example:

```powershell
python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\train_objective.py `
  --h5 datasets\LIDC\lidc_patches.h5 `
  --split-file LungInsight_v2\backend\models\nodule_ranker_3d\checkpoints_model2_final\split.json `
  --out-dir LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints `
  --objective ordinal_reader `
  --rank-loss-weight 0.5 `
  --ranking-target reader_pairwise `
  --ranking-min-confidence 0.10 `
  --device auto `
  --batch-size 8 `
  --num-workers 4 `
  --epochs 75 `
  --patience 10 `
  --seed 123
```

Runs are isolated under `rank_<weight>_reader_pairwise_pair_<threshold>_min_conf_<confidence>` so they do not overwrite legacy runs or another confidence setting.

Migration: Existing `mean`, `median`, and `mean_median_blend` commands, output paths, checkpoints, and loss semantics are unchanged. `--ranking-min-confidence` is only used by `reader_pairwise`; its default does not affect old runs. The evaluator continues to read old checkpoints, defaulting a missing ranking target to `mean`.

## Multi-task versus single-task comparison

Use `--attribute all` (the default) for the existing joint objective and `--attribute <name>` to train only one ordered target. In a single-task run, only that target contributes to supervised loss, ranking loss, validation metrics, and checkpoint selection. The other output heads remain in the forward pass, but are excluded from loss and metrics. Best-checkpoint selection is `<attribute>.spearman_ge3`; if it is undefined for the entire run, selection falls back to minimum validation loss and records that fallback. Single-task runs log the current/best epoch, Spearman values, prediction standard deviation, and SD ratio.

## Frozen hybrid groups

`--attribute-group` selects the attributes that contribute to ordinal loss, ranking loss, validation metrics, and checkpoint selection. Choices are `all` (legacy behavior), `final_hybrid` (the six ordered attributes), `lobulation_spiculation`, `subtlety`, `sphericity`, `margin`, and `texture`. `all` retains its legacy implicit `mean` ranking target; named hybrid groups default to the frozen `median` target, rank-loss weight `0.5`, and pair threshold `0.5`. The objective remains explicitly required and should be `ordinal_reader` for the final hybrid models.

Train the five final models:

```powershell
$groups = @("lobulation_spiculation", "subtlety", "sphericity", "margin", "texture")
foreach ($group in $groups) {
  python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\train_objective.py `
    --h5 datasets\LIDC\lidc_patches.h5 `
    --split-file LungInsight_v2\backend\models\nodule_ranker_3d\checkpoints_model2_final\split.json `
    --out-dir LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints `
    --objective ordinal_reader `
    --attribute-group $group `
    --ranking-target median `
    --rank-loss-weight 0.5 `
    --ranking-pair-threshold 0.5 `
    --device auto `
    --batch-size 8 `
    --num-workers 4 `
    --epochs 75 `
    --patience 10 `
    --seed 123
}
```

Each run writes `best.pth`, `run.json`, and `history.csv` under `checkpoints\hybrid\<attribute-group>\`. Lobulation/spiculation selection averages both `spearman_ge3` metrics and falls back to validation loss if either is undefined; each single-attribute model selects its own `spearman_ge3`.

`infer_hybrid.py` accepts one raw-HU NPY patch shaped `[64,64,64]` (or a single-item batch) and the five checkpoints, then writes the six expected-rating predictions. `evaluate_hybrid.py` accepts those checkpoints plus `--h5` and `--split-file`, combines their outputs, and reports `M6>=3`, `M6all`, and per-attribute metrics on `val` or final `test`.

Run the matched five-seed comparison with the champion objective and ranking configuration:

```powershell
$seeds = @(123, 456, 789, 1337, 2025)
$attributes = @("all", "subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")
foreach ($seed in $seeds) {
  foreach ($attribute in $attributes) {
    python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\train_objective.py `
      --h5 datasets\LIDC\lidc_patches.h5 `
      --split-file LungInsight_v2\backend\models\nodule_ranker_3d\checkpoints_model2_final\split.json `
      --out-dir LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints `
      --objective ordinal_reader `
      --ranking-target median `
      --rank-loss-weight 0.5 `
      --ranking-pair-threshold 0.5 `
      --attribute $attribute `
      --device auto `
      --batch-size 8 `
      --num-workers 4 `
      --epochs 75 `
      --patience 10 `
      --seed $seed
  }
}
```

`all` retains the existing checkpoint path. Single-task checkpoints append `_attribute_<name>`, so they cannot overwrite multi-task runs. Each checkpoint includes its selected attribute, selection metric, seed, and validation metrics.

Analyze the matched runs:

```powershell
python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\analyze_multitask_vs_singletask.py `
  --checkpoint-dir LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints\ordinal_reader `
  --output LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints\multitask_vs_singletask.csv `
  --objective ordinal_reader `
  --ranking-target median `
  --rank-loss-weight 0.5 `
  --ranking-pair-threshold 0.5
```

The CSV has one row per ordered attribute: `attribute`, `multitask_best`, `singletask_best`, `multitask_sd_ratio`, `singletask_sd_ratio`, and `improvement`. Performance and SD ratios are means over matched seeds; `improvement = singletask_best - multitask_best`, so positive values favor single-task training. The analyzer rejects duplicate checkpoints, incomplete attribute sets, non-finite ranking scores, or ambiguous experiment configurations rather than mixing runs. The split, TEST isolation, backbone, encoder, attention, and objective/loss definitions are unchanged for `all`.

## Fixed architecture

`64^3` input → Conv3D/GN/GELU stem → residual stages → `128×8×8×8` → 512 spatial tokens → task-specific query attention with 96-D features → eight attribute heads.

No augmentation. No reconstruction-pretrained initialization. No cross-attribute attention. No TEST use during training.

## Training

From the project root:

### Objective A

```powershell
python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\train_objective.py `
  --h5 datasets\LIDC\lidc_patches.h5 `
  --split-file LungInsight_v2\backend\models\nodule_ranker_3d\checkpoints_model2_final\split.json `
  --out-dir LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints `
  --objective ordinal_reader `
  --rank-loss-weight 0.25 `
  --ranking-target mean `
  --ranking-pair-threshold 0.5 `
  --device auto `
  --batch-size 8 `
  --num-workers 4 `
  --epochs 75 `
  --patience 10 `
  --seed 123
```

### Objective B

```powershell
python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\train_objective.py `
  --h5 datasets\LIDC\lidc_patches.h5 `
  --split-file LungInsight_v2\backend\models\nodule_ranker_3d\checkpoints_model2_final\split.json `
  --out-dir LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints `
  --objective mean_regression `
  --rank-loss-weight 0.25 `
  --ranking-target mean `
  --ranking-pair-threshold 0.5 `
  --device auto `
  --batch-size 8 `
  --num-workers 4 `
  --epochs 75 `
  --patience 10 `
  --seed 123
```

Each run is saved under `checkpoints/<objective>/seed_<seed>/rank_<weight>_<target>_pair_<threshold>/`, so sweep combinations do not overwrite one another. For the recommended six-run sweep, repeat both weight values (`0.25`, `0.50`) with each target (`mean`, `median`, `mean_median_blend`).

## Compare validation runs

```powershell
python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\compare_runs.py `
  --objective-a LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints\ordinal_reader\seed_123\rank_0.25_mean_pair_0.5\run.json `
  --objective-b LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints\mean_regression\seed_123\rank_0.25_mean_pair_0.5\run.json
```

## Validation evaluation

```powershell
python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\evaluate_objective.py `
  --checkpoint path\to\best.pth `
  --h5 datasets\LIDC\lidc_patches.h5 `
  --split-file LungInsight_v2\backend\models\nodule_ranker_3d\checkpoints_model2_final\split.json `
  --split val `
  --device auto
```

## Final TEST evaluation

Only after the architecture/objective is frozen:

```powershell
python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\evaluate_objective.py `
  --checkpoint path\to\best.pth `
  --h5 datasets\LIDC\lidc_patches.h5 `
  --split-file LungInsight_v2\backend\models\nodule_ranker_3d\checkpoints_model2_final\split.json `
  --split test `
  --device auto `
  --output final_test_report.json
```

## NPY inference

The deployed model does not require the H5. NPY input is raw HU `64×64×64`, or a batch `[N,64,64,64]` / `[N,1,64,64,64]`.

```powershell
python LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\infer_npy.py `
  --checkpoint path\to\best.pth `
  --input-npy path\to\detected_nodule_patches.npy `
  --output-json nodule_predictions.json `
  --device auto
```

## Selection metric

Primary: M6>=3 = mean Spearman across the six ordered attributes on validation nodules with at least three valid readers.

Secondary: M6all, per-attribute Spearman/MAE/RMSE/SD ratio, and nominal-task NLL/accuracy. Calcification also reports present-vs-absent AUROC, where “present” means categories 1–5 and “absent” means category 6.

NaN correlations are never treated as zero. The trainer records the number of finite ordered correlations. If all six ordered correlations are undefined, checkpoint selection falls back to minimum validation loss with `selection_metric=-val_loss_fallback` clearly recorded.

## Data-safety invariants

- The fixed H5 is never modified.
- The canonical split is never regenerated or changed.
- Training loads only TRAIN+VAL CT patches and reader labels; TEST CT/labels are not loaded by the trainer.
- HDF5 fancy indices are sorted before indexed reads and restored to the requested order.
- Missing/non-finite reader ratings are filtered explicitly; `NaN*0` masking is never used.
- The known invalid `internalStructure=5` rating is excluded only from that affected attribute/reader.
