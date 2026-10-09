# Milestone 2 Object Detection with YOLOv8

Train a 73-class YOLOv8s detector using synthetic multi-object images made from the finalized shared single-object dataset. Predictions export the final Object IDs, names, confidence scores, and pixel bounding boxes. The shared raw dataset is read-only.

## Dataset and annotations

- Final dataset: https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS
- Final Object IDs: https://docs.google.com/spreadsheets/d/1OCBWlTK8JSzrgUxON3thJWAKRKU2Pi-V9Ssn2BbDxds/edit#gid=0
- `class_mapping.json` contains the 73 official IDs and names. YOLO class indices are separate zero-based integers, with the exact mapping preserved.
- `annotations/source_annotations.json` contains 365 visually annotated source photos: three train, one validation, and one test photo per class. Annotation coordinates are normalized xyxy on EXIF-oriented primary RGB frame 0. Multiple visible target instances have separate boxes. Annotation review was performed by an assistant; no independent human verification is claimed.
- The source pool uses the existing seed-42, exact-deduplicated Milestone 1 manifests. Photos tagged `_bg_` are ineligible for this *selected detection annotation subset*. Ten initial candidates were replaced because the target was absent or too small for a reliable annotation. These decisions do not delete or alter raw images. See `annotations/annotation_exclusions.json`.
- Generate **1,460** 640x640 multi-object images: **1,168 train / 146 validation / 146 test**, an exact 80/10/10 image split with seed 42. Sources are separated *before* synthesis, preventing one source photo from appearing in different composite splits. Every split covers all 73 IDs.
- Each image combines 2–4 randomly selected distinct classes. Tight rectangular crops preserve 5–15% contextual margins, are resized and randomly placed without overlap, and retain every annotated instance. The crops retain their internal background; they are not segmentation masks. Training crops also receive mild brightness/contrast changes and horizontal flips.
- Source checksums, generated image/label checksums, transforms, labels, and paths are recorded. The validator checks all labels, official IDs, split coverage, and source disjointness.

## Reproduce

Generate the frozen JPEGs with Python 3.12 and the separate generation environment below. Keep the original filenames and folder layout beneath `--raw-root`, for example `images_OBJ001/OBJ001_045.jpg`; the annotation archive specifies every required relative path and SHA-256 hash. Download and extract the final dataset locally without changing the shared Drive files.

```bash
python3.12 -m venv .venv-generation
.venv-generation/bin/python -m pip install -r generation-requirements.txt
.venv-generation/bin/python -m unittest discover -s tests -v
.venv-generation/bin/python generate_dataset.py --raw-root /path/to/final-local/images
.venv-generation/bin/python validate_dataset.py data/synthetic_seed42
```

The generator refuses to overwrite an existing dataset. The archived source annotations are the reproducible input to synthesis; manual visual decisions are recorded rather than recomputed automatically. The historical `select_sources.py` and `finalize_annotations.py` review helpers depend on the original Milestone 1 workspace and are not needed to reproduce this finalized annotation set. The seeded generation test also uses that local source pool when available, otherwise it skips. Other geometry/matching tests do not need the raw dataset.

For local training/inference, use a separate environment with `requirements.txt` and compatible PyTorch/Torchvision. Do not install the generation and training pins into the same environment. The completed Colab run used Python 3.13.15, PyTorch 2.11.0+cu130, CUDA 13.0 and Ultralytics 8.3.221. Colab's matching CUDA installation is preserved by the worker; its full package freeze is in `reports/environment.json`. Generation used Python 3.12.14, Pillow 12.3.0 and NumPy 2.3.5, recorded in `annotations/generation_environment.json`. Same seed alone does not guarantee identical JPEG bytes across different image-library versions or identical GPU results across hardware/software stacks.

```bash
python -m pip install -r requirements.txt
python train_detector.py --dataset data/synthetic_seed42 --output runs --epochs 50 --device 0
```

## Colab CLI

With the authenticated `colab` CLI installed:

```bash
.venv-generation/bin/python run_colab.py --session milestone2-yolov8s-YOUR_UNIQUE_NAME --gpu A100
```

The launcher verifies the local dataset, creates an owned GPU session, uploads a checksum-verified archive in bounded parts, runs tests and the full dataset validator, trains and evaluates, retrieves SHA-256-verified artifacts, then releases the GPU. If artifact retrieval fails, it preserves the runtime and prints recovery instructions. Recover with the same session/config/output and `--recover`. `--attach` is reserved for an explicitly selected, already-created task session.

## Training and evaluation protocol

- Required detector: COCO-pretrained **YOLOv8s** (`yolov8s.pt`), Ultralytics 8.3.221.
- Budget: 50 epochs, input 640, batch 32, seed 42, no early stopping.
- AdamW, initial LR 0.001, cosine schedule with final factor 0.01, weight decay 0.0005, three warm-up epochs.
- Training-only YOLO mosaic (closed for the final ten epochs), geometric and HSV augmentation. Exact resolved arguments and installed packages are archived.
- RGB input tensors are scaled to 0–1 by division by 255 in the detector's standard preprocessing.
- Select `best.pt` by **validation mAP50–95**, using the library's saved-checkpoint rule. The actual save epoch is recorded by a callback; the test set is excluded from selection.
- Evaluate validation and test separately: precision, recall, mAP50, mAP50–95, and per-class AP.
- Select a confidence threshold on validation micro-F1 with class-aware matching at IoU >= 0.5, then fix it for test. Also report TP/FP/FN and the fraction of test images with *every* annotated object correctly detected and no false positives.
- Six gallery paths are sampled with seed 42 independently of results. Complete per-image detections are saved; galleries never substitute for aggregate metrics.

## Run detection on an image

```bash
python predict.py --weights artifacts/yolov8s_seed42/experiment/runs/yolov8s_seed42/weights/best.pt \
  --source /path/to/multi_object_image.jpg --device cpu \
  --threshold-file artifacts/yolov8s_seed42/experiment/runs/operating_threshold.json \
  --output predictions
```

The annotated output and `detections.json` show Object IDs and locations. JSON boxes are `(x1,y1,x2,y2)` in original-image pixels, with the origin at the upper-left. Confidence is a model score rather than a calibrated correctness probability.

## Limits

The benchmark consists of synthetic rectangular photo patches, with only three distinct train and one validation/test source image per class. Repeated crops within a split do not create independent source photos. Exact hashes are disjoint, but near-duplicate captures, the same physical objects, or shared backgrounds may cross source splits. Images retain context and some partial targets; annotation precision and incidental background content may affect learning. Test metrics establish performance on this fixed synthesis distribution and do not establish performance on independently photographed real multi-object scenes. Collect and annotate such a test set before claiming real-world generalization.

## References

- https://docs.ultralytics.com/models/yolov8/
- https://docs.ultralytics.com/datasets/detect/
- https://docs.ultralytics.com/modes/train/

## Results and report

Completed 50 epochs on a Colab A100. The best saved checkpoint is epoch **30**, selected using validation mAP50–95. The final checkpoint was reevaluated separately on validation and test.

| Split | Mean precision | Mean recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Validation | 79.77% | 75.32% | 83.57% | 69.45% |
| Test | 80.49% | 61.54% | 71.31% | 57.85% |

The library's precision/recall are class means at the best smoothed mean-F1 point on each evaluated split's curve; they are descriptive curve summaries. The separate operational threshold **0.50** was chosen only on validation micro-F1, then fixed for test. At that threshold, test precision is **78.34%**, recall **59.73%**, F1 **0.6778**, with **264 TP / 73 FP / 178 FN**. Every annotated object was correctly detected with no false positives in **31/146 images (21.23%)**, so this model does not reliably find every object in every image.

- [Word report with plots and detection examples](reports/milestone2_report.docx)
- [Full class-level validation/test results](reports/per_class_metrics.csv)
- [Fixed-threshold metrics for every test image](reports/test_operating_metrics.json)
- [Saved experiment evidence](reports/evidence/), including full predictions, generated manifest, training history and GPU-release receipt

Generated images and model weights remain outside Git. In the original workspace the verified download is `artifacts/yolov8s_seed42/experiment`, with the selected model at `runs/yolov8s_seed42/weights/best.pt`. Its SHA-256 is `183ddf6200b3ba845560e2d7b94c03386c82b4a8ba0ca398ab15ba95b5ca9211`. A clone can regenerate data and retrain; it does not already contain weights.

Verify the saved reported metrics without a GPU or source photos:

```bash
python verify_saved_results.py
```

To regenerate the scientific figures and Word report from the full local experiment, install `report-requirements.txt` in a separate environment, then run:

```bash
python analyze_results.py --experiment artifacts/yolov8s_seed42/experiment
python render_word_report.py
```

The Word file was rendered and every page visually checked before delivery. The generated images and labels passed all validation checks; the Colab upload/download parts passed SHA-256 verification, and the dedicated GPU session was stopped after retrieval. Results apply to the synthetic benchmark; no real-scene evaluation is claimed.
