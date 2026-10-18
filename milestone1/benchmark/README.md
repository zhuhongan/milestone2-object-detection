# Controlled group-model benchmark

This is the controlled four-model classification experiment. See the
[Milestone 1 guide](../README.md) for the code map, portable dataset paths,
notebooks and commands to run from a repository clone. Original group files and
the locked raw dataset are not modified or required in Git.

## Models and protocol

The four backbones are VGG16, ResNet50, MobileNetV2 with width multiplier 1.4, and
ConvNeXt-Tiny. Each starts from ImageNet-1K weights and uses a shared classifier:
pooled backbone features -> Linear(256) -> LayerNorm -> GELU -> Dropout(0.3) ->
Linear(73). The features and pretrained normalization layers remain specific to
the backbone. The original VGG flattened feature head is replaced with global
average pooling to standardize the downstream classifier.

All four models use the exact same audited 80/10/10 seed-42 manifests from the
earlier experiment: 5,851 training, 735 validation and 729 test images. This is a
reused fixed test split, not a new independent test collection. No arbitrary
filename-based background exclusion or per-folder cap is applied.

- Exactly **20 epochs**: 5 frozen-backbone epochs + 15 full fine-tuning epochs.
- Fine-tuning continues from epoch 5; the optimizer resets at the transition.
- No early stopping or per-model hyperparameter search.
- Shared 224-pixel training crops, flip/color augmentation, and ImageNet normalization.
- Shared evaluation geometry: resize shorter side to 256, center crop to 224.
- Batch size 32, AdamW, weight decay 0.01, gradient clipping at 1.0.
- Frozen-head LR 0.001; fine-tuning backbone/head LRs 0.00001/0.0001.
- Cosine decay within each phase. Seeds and augmentation/shuffle streams are
  deterministic functions of base seed 42 and epoch.
- Best checkpoint and winning model selected by validation macro-F1, then lower loss.
- Test evaluation begins only after all four models finish their 20 epochs and
  the validation-selected winner has been saved.

The PyTorch model families are standardized adaptations of the submitted Keras
scripts. Torchvision ResNet50 uses the V1.5 variant, and MobileNetV2 uses timm's
`mobilenetv2_140.ra_in1k` weights. The exact identities are saved in `config.json`.
Pretraining recipes differ, so this is a controlled downstream comparison of
model-and-weight configurations, not an isolation of architecture alone.

`source_inventory.json` records the inspected files, their SHA-256 hashes, and
the differences from the original scripts. The originals are never executed by
this pipeline.

## Run using Colab CLI

From the `milestone1/` project directory, after acquiring the local raw snapshot,
with the authenticated Colab CLI:

```bash
python3 benchmark/run_colab_benchmark.py \
  --session UNIQUE_SESSION_NAME \
  --output artifacts/NEW_BENCHMARK_RUN \
  --prepared-dir prepared --raw-snapshot data/raw/final-local.tar.gz \
  --gpu A100
```

The launcher allocates its own GPU, uploads code, exact prepared manifests and
the verified local raw snapshot, runs behavioral GPU checks, trains all models,
retrieves SHA-256-verified artifact parts, and stops its session. Existing sessions
are not used or changed. The full Google Sheet and unrelated student columns are
not uploaded.

If transfer fails after training, retry retrieval with the same output/session:

```bash
python3 benchmark/run_colab_benchmark.py \
  --session EXISTING_SESSION_NAME --output artifacts/EXISTING_BENCHMARK_RUN --recover
```

While remote training runs, `colab exec` streams epoch logs. The dedicated session
configuration is `.colab-benchmark-sessions.json`; do not publish it. The runtime
may remain billable until export recovery succeeds and the launcher stops it.

## Outputs and report

The extracted experiment contains `benchmark_run/config.json`, `selection.json`,
`comparison.csv`, environment/package records, and for each model:

- `history.csv/json`: all 20 epochs, validation metrics, timings and update counts.
- `best.pt`: validation-selected checkpoint with official IDs and model name.
- `last.pt`: final epoch with optimizer/scaler state for diagnosis or resumption.
- Validation/test metrics, per-class scores, confusion matrices and predictions.
- Completion markers tying each evaluation to its checkpoint hash.

Create plots and the editable report with a Python environment containing NumPy,
Matplotlib and Pillow, then render the PDF with ReportLab and Pillow:

```bash
python benchmark/analyze_benchmark.py \
  --experiment artifacts/group_benchmark_20ep_seed42/experiment \
  --raw-root data/raw/final-local/images --output reports/group_benchmark
python benchmark/render_report.py \
  --data reports/group_benchmark/report_data.json \
  --output output/pdf/milestone1_group_comparison.pdf
```

The analysis verifies full epoch budgets, manifest and checkpoint hashes, official
labels, and accuracy/macro-F1 recomputed from predictions. Shared examples are
sampled independently of model results using seed 42. The failure gallery is
explicitly selected by highest-confidence errors. All illustrated paths and
selection rules are saved. Synthetic smoke-test results are never reported as
dataset performance.
