# Milestone 1 — object classification

Compare **VGG16, ResNet50, MobileNetV2 (width 1.4) and ConvNeXt-Tiny** on the finalized 73-class dataset. All four models use the same frozen **80/10/10 split, seed 42**, and complete **20 epochs: 5 frozen + 15 full fine-tuning**, without early stopping. The selected model is **ConvNeXt-Tiny, epoch 15**, chosen by validation macro-F1 before test evaluation.

The main comparison entry point is [`benchmark/train_benchmark.py`](benchmark/train_benchmark.py). [`train.py`](train.py) supplies shared data, metric and training helpers; its standalone CLI also preserves the earlier ConvNeXt pilot, which used a different head and stopping protocol. Use the benchmark entry point for the four-model experiment.

## Code map for the presentation

| Part | Main files | What to explain |
|---|---|---|
| Final IDs and data acquisition | [build_mapping.py](build_mapping.py), [download_dataset.py](download_dataset.py), [metadata/object_mapping.csv](metadata/object_mapping.csv) | Final folder-to-ID/name mapping, paginated read-only Drive download and checksum checks. |
| Validation, deduplication and splits | [prepare_dataset.py](prepare_dataset.py), [prepared/](prepared/) | Image decoding, EXIF/RGB, primary MPO frame, exact duplicates, class-stratified 80/10/10 allocation and source separation. |
| Four architectures and common training | [benchmark/train_benchmark.py](benchmark/train_benchmark.py), [train.py](train.py) | The four `Classifier` branches, common classifier head, augmentation, frozen/fine-tuning phases and validation selection. |
| Colab execution and recovery | [benchmark/run_colab_benchmark.py](benchmark/run_colab_benchmark.py), [benchmark/colab_worker.py](benchmark/colab_worker.py), [run_colab.py](run_colab.py) | Dedicated session, verified upload/export, resume and GPU shutdown. |
| Results, plots and reports | [benchmark/analyze_benchmark.py](benchmark/analyze_benchmark.py), [benchmark/render_report.py](benchmark/render_report.py), [benchmark/render_word_report.py](benchmark/render_word_report.py) | Recomputed metrics, model comparisons, learning curves, per-class errors, examples and final justification. |
| Live single-object identification | [notebooks/milestone1_live_demo.ipynb](notebooks/milestone1_live_demo.ipynb), [demo/](demo/) | Matching inference preprocessing, loading the exact checkpoint and exporting final predicted IDs/names. |

The files are organized by pipeline stage; the table suggests presentation responsibilities, not personal authorship.

## Dataset and fixed protocol

- [Final locked dataset](https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS)
- [Final Object ID sheet](https://docs.google.com/spreadsheets/d/1OCBWlTK8JSzrgUxON3thJWAKRKU2Pi-V9Ssn2BbDxds/edit#gid=0)
- 7,343 valid photos, 28 exact duplicate copies excluded before splitting, 7,315 retained sources.
- **5,851 train / 735 validation / 729 test**. Small departures from exact global percentages come from per-class integer allocation; every class appears in all splits.
- No background-filename exclusion or per-folder image cap in classification. Milestone 2's positive-target filtering is a separate detection preprocessing policy.
- EXIF-oriented RGB, primary frame 0 for JPEG/MPO. Training uses random resized 224-pixel crops, horizontal flips and mild color jitter. Evaluation resizes the shorter side to 256 and centre-crops to 224. ImageNet mean/std are fixed for both.
- Shared head: pooled backbone features → Linear(256) → LayerNorm → GELU → Dropout(0.3) → Linear(73).
- Batch 32, AdamW, weight decay 0.01, cosine decay within each phase and gradient clipping 1.0. Frozen-head LR 0.001; fine-tuning backbone/head LRs 0.00001/0.0001.
- Best checkpoint and model: validation macro-F1, then lower validation loss. All four runs finish before test evaluation begins.

[`prepared/`](prepared/) contains the exact saved manifests, audit, class mapping and exclusions, with no image bytes. Original audit paths are historical provenance; training takes `--raw-root` explicitly and verifies actual bytes against the frozen manifests. CSV line endings are preserved in Git because their hashes are part of the audit. The three mapping columns are included; the full sheet snapshot and unrelated student metadata are not published.

The model configurations are standardized PyTorch adaptations of the group submissions. ResNet50 uses Torchvision's V1.5 implementation and MobileNetV2 uses timm's `mobilenetv2_140.ra_in1k` pretrained weights. Pretraining recipes differ, so this compares architecture-and-weight configurations under a common downstream protocol. [source_inventory.json](benchmark/source_inventory.json) records the adaptations and original file hashes; the original Keras notebooks/HTML are not needed or executed by this pipeline.

## Run in Colab

- **Training and full comparison:** [milestone1_colab.ipynb](notebooks/milestone1_colab.ipynb). Upload to Colab, choose a CUDA GPU runtime and run in order. It contains its own code/frozen metadata and downloads verified finalized images. It trains all four models and creates fresh metrics, plots and a Word report.
- **Live demo with file uploads:** [milestone1_live_demo.ipynb](notebooks/milestone1_live_demo.ipynb). Supply the selected `.pt` checkpoint, then upload test photos. No retraining or dataset download is needed.
- **Drive input/output folders:** [Milestone1_Drive_Demo.ipynb](notebooks/Milestone1_Drive_Demo.ipynb). Reads `MyDrive/Assignment_Live_Demo/milestone1/input`, loads the checkpoint from `models/` under the same root and saves to `milestone1/output`. Each teammate must have that folder/checkpoint in their own mounted Drive or configure the root for their account.

The training notebook bundles the frozen experiment sources for portability. Editable `.py` files are supplied alongside it for code review. Checkpoints are not included in Git. The live demo checks the exact selected model hash; a separately retrained checkpoint requires its own verified mapping and inference configuration. CPU inference works; the full training comparison expects CUDA. GPU/software changes can change numerical results despite a fixed seed.

## Run from a clone using Colab CLI

Run commands from **this `milestone1/` directory**, with an already authenticated Colab CLI. Install dependencies into an isolated environment, preserving a matching PyTorch/torchvision CUDA pair on GPU machines:

```bash
python -m pip install -r requirements.txt
python download_dataset.py --dest data/raw/final
tar -czf data/raw/final-local.tar.gz -C data/raw final
python benchmark/run_colab_benchmark.py \
  --session YOUR_UNIQUE_SESSION_NAME --gpu A100 \
  --prepared-dir prepared --raw-snapshot data/raw/final-local.tar.gz \
  --output artifacts/group_benchmark_20ep_seed42
```

Acquisition reads the finalized dataset without modifying it. If public Drive access is unavailable, use `--auth adc` with your own read-only Application Default Credentials. The launcher rejects missing snapshot/manifest paths before allocating a GPU. The snapshot layout must contain `final/images` and its acquisition manifest, as created by the commands above.

Use the checked-in frozen manifests for the same experiment. To independently validate a local snapshot and generate manifests in a separate output folder:

```bash
python prepare_dataset.py --raw-root data/raw/final/images \
  --mapping metadata/object_mapping.csv --output prepared_check --seed 42
```

Never overwrite or edit the locked raw data. If artifact retrieval fails, retain the same output/session and recover:

```bash
python benchmark/run_colab_benchmark.py --session YOUR_UNIQUE_SESSION_NAME \
  --output artifacts/group_benchmark_20ep_seed42 --recover
```

The launcher stops its dedicated GPU after verified export. Failed export can retain the runtime for recovery; session registration files are private and ignored by Git. The shared transfer implementation is in `run_colab.py`; the controlled comparison launcher is in `benchmark/`.

## Plots and reports

After training/export, install `report-requirements.txt` and run:

```bash
python -m pip install -r report-requirements.txt
python benchmark/analyze_benchmark.py \
  --experiment artifacts/group_benchmark_20ep_seed42/experiment \
  --raw-root data/raw/final/images --output reports/group_benchmark
python benchmark/render_report.py --data reports/group_benchmark/report_data.json \
  --output output/pdf/milestone1_group_comparison.pdf
python benchmark/render_word_report.py
```

The Word renderer uses `reports/group_benchmark/report_data.json` and saves `output/docx/milestone1_group_comparison.docx`. Review generated reports and group details before submission. `build_report.py` is the preserved report generator for the earlier single-backbone pilot; the current four-model report generators are in `benchmark/`.

The completed experiment's compact [comparison.csv](recorded_results/comparison.csv) is included as a recorded result, not generated by installation or smoke tests:

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| VGG16 | 0.9494 | 94.92% | 0.9493 |
| ResNet50 | 0.9558 | 95.34% | 0.9533 |
| MobileNetV2 (1.4) | 0.9574 | 93.83% | 0.9388 |
| **ConvNeXt-Tiny** | **0.9660** | **96.57%** | **0.9659** |

The test split is reused from the earlier pilot; it is a fixed benchmark rather than a new independent test collection. Exact deduplication does not prove independence between near captures or shared physical objects/backgrounds.

## Checks

```bash
python -m unittest test_prepare_dataset test_download_dataset test_training benchmark.test_benchmark -v
python -m unittest discover -s tests -v
```

Actual CPU integration for the earlier ConvNeXt training helper is opt-in with `RUN_TRAINING_SMOKE=1 SMOKE_DEVICE=cpu`. Four-backbone GPU behavioral tests run when CUDA is available and skip explicitly on CPU. These synthetic checks are not dataset performance results. All code, notebook and metadata checks can run without private session configuration or published model weights.
