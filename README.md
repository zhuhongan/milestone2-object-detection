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

Use Python 3.10 or newer. Install the pinned detector package and compatible PyTorch/Torchvision. Colab's matching CUDA installation is preserved.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python generate_dataset.py --raw-root /path/to/final-local/images
python validate_dataset.py data/synthetic_seed42
python train_detector.py --dataset data/synthetic_seed42 --output runs --epochs 50 --device 0
```

The generator refuses to overwrite an existing dataset. The source file paths and SHA-256 hashes must match the finalized snapshot. The archived source annotations are the reproducible input to synthesis; manual visual decisions are recorded rather than recomputed automatically.

## Colab CLI

With the authenticated `colab` CLI installed:

```bash
python run_colab.py --session milestone2-yolov8s-YOUR_UNIQUE_NAME --gpu A100
```

The launcher verifies the local dataset, creates an owned GPU session, uploads a checksum-verified archive in bounded parts, runs tests and the full dataset validator, trains and evaluates, retrieves SHA-256-verified artifacts, then releases the GPU. If artifact retrieval fails, it preserves the runtime and prints recovery instructions. Recover with the same session/config/output and `--recover`. `--attach` is reserved for an explicitly selected, already-created task session.

## Training and evaluation protocol

- Required detector: COCO-pretrained **YOLOv8s** (`yolov8s.pt`), Ultralytics 8.3.221.
- Budget: 50 epochs, input 640, batch 32, seed 42, no early stopping.
- AdamW, initial LR 0.001, cosine schedule with final factor 0.01, weight decay 0.0005, three warm-up epochs.
- Training-only YOLO mosaic (closed for the final ten epochs), geometric and HSV augmentation. Exact resolved arguments and installed packages are archived.
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

The completed experiment and report are added after verified GPU training and held-out evaluation. Generated data and model weights are kept outside Git; this repository contains the code, mappings, reviewed annotations, and report.
