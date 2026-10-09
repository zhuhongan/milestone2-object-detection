# Milestone 2 Object Detection with YOLOv8

Train a 73-class YOLOv8s detector using synthetic multi-object images made from the finalized shared single-object dataset. Predictions export the final Object IDs, names, confidence scores, and pixel bounding boxes. The shared raw dataset is read-only.

## Experiment overview

- Final dataset: https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS
- Final Object IDs: https://docs.google.com/spreadsheets/d/1OCBWlTK8JSzrgUxON3thJWAKRKU2Pi-V9Ssn2BbDxds/edit#gid=0
- `class_mapping.json` preserves the mapping between the 73 final Object IDs and zero-based YOLO class indices.

## Dataset and annotations

All **7,343 finalized photos** have annotation records and YOLO label files in `annotations/all_photos_seed42/`. The 28 exact duplicate copies inherit their canonical photo’s annotation and split; synthesis uses unique sources only. Every one of the 7,315 unique photos was inspected on 151 contact sheets, with 1,379 explicit corrections or exclusions recorded in `annotations/all_photo_review_decisions.json`. Model proposals were produced with pinned Grounding DINO on a Colab A100, then visually checked and corrected by the assistant. There is no independent human annotation review.

There are **6,933 unique photos with a visible assigned target**. The other **382 unique photos** have no identifiable visible target of their folder’s Object ID. They remain in the annotation inventory with an exclusion reason. A target-absent record does **not** establish the absence of every other Object ID; do not use its empty label as a universally annotated negative image. Incidental objects from other classes have not been exhaustively labelled. `_bg_` filenames were inspected rather than automatically excluded.

The dataset contains **10,000 composites: 8,000 train / 1,000 validation / 1,000 test**, with seed 42. Seeded per-class cycles guarantee that every eligible positive source is used. The deduplicated source splits are preserved before synthesis. Source photos retain their original dimensions; EXIF-oriented RGB images are cropped and resized during composition. The shared and local raw photos are unchanged.

```bash
python generate_dataset.py --raw-root /path/to/final-local/images \
  --annotations annotations/all_photos_seed42/positive_source_annotations.json \
  --output data/synthetic_all_photos_seed42 --counts 8000 1000 1000 \
  --cover-sources --seed 42
python validate_dataset.py data/synthetic_all_photos_seed42
```

Train the detector through Colab CLI:

```bash
python run_colab.py --session milestone2-yolov8s-all-photos-YOUR_UNIQUE_NAME \
  --gpu A100 --dataset data/synthetic_all_photos_seed42 \
  --output artifacts/yolov8s_all_photos_seed42
```

The model completed **50 epochs on a Colab A100**. Validation selected epoch **50**. On the 1,000 test composites, **mAP50 is 98.65% and mAP50–95 is 94.26%**. The validation-selected confidence threshold is **0.65**; test micro precision is **98.77%**, recall **97.80%**, F1 **0.9829**, and every annotated object is correctly detected with no false positives in **929/1,000 images (92.90%)**.

- [Updated Word report with new plots and detection examples](reports/all_photos_seed42/milestone2_report.docx)
- [Experiment results and reproducible evidence](reports/all_photos_seed42/)
- Selected weights in the local workspace: `artifacts/yolov8s_all_photos_seed42/experiment/runs/yolov8s_seed42/weights/best.pt`
- Selected checkpoint SHA-256: `a2c56355ab06cc473390c2bcc618bd5af71cebba7d8dd2e7a5d2d7410c4a161c`

Results describe the synthetic benchmark. Accuracy on independently photographed real multi-object scenes has not been measured.

```bash
python predict.py --weights artifacts/yolov8s_all_photos_seed42/experiment/runs/yolov8s_seed42/weights/best.pt \
  --source /path/to/multi_object_image.jpg --device cpu \
  --threshold-file reports/all_photos_seed42/operating_threshold.json \
  --output predictions_all_photos
python verify_saved_results.py --reports reports/all_photos_seed42
```

Rebuild the report from the verified experiment and frozen images:

```bash
python analyze_results.py --experiment artifacts/yolov8s_all_photos_seed42/experiment \
  --dataset data/synthetic_all_photos_seed42 \
  --annotations annotations/all_photos_seed42/positive_source_annotations.json \
  --annotation-summary annotations/all_photos_seed42/annotation_summary.json \
  --output reports/all_photos_seed42
python render_word_report.py --data reports/all_photos_seed42/report_data.json \
  --output reports/all_photos_seed42/milestone2_report.docx
```

Use the separate report environment in `report-requirements.txt` for figure and Word generation.

For fresh model-assisted proposals, `inventory_all_sources.py` reads the finalized Milestone 1 manifests, `run_annotation_colab.py` runs the GPU proposal job, and `review_all_annotations.py prepare` creates review sheets. The archived reviewed JSON annotations are sufficient for deterministic synthesis; rerunning proposals cannot replace recorded visual review decisions. Proposal-model revision, environment and GPU-release evidence are archived with the annotations.

## Reproducible environments

Use Python 3.12 and `generation-requirements.txt` in a separate environment to generate and validate images. Preserve the raw folder layout and source filenames listed in the annotation JSON. The generator refuses to overwrite an existing dataset. Run `python -m unittest discover -s tests -v` for geometry, matching, coverage and reproducibility checks.

Training and inference use `requirements.txt` and compatible PyTorch/Torchvision in a separate environment. The completed Colab run used Python 3.13.15, PyTorch 2.11.0+cu130, CUDA 13.0 and Ultralytics 8.3.221. Package versions and resolved parameters are archived in `reports/all_photos_seed42/`. Matching seeds alone do not guarantee identical results across different hardware or software versions.

The Colab launcher validates the dataset, uploads a checksum-verified archive in bounded parts, trains and evaluates, retrieves verified artifacts and releases the GPU. If retrieval fails, recover with the same session/config/output and `--recover`; the launcher preserves the runtime for recovery.

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

## Interpretation and saved artifacts

Source paths, byte hashes and RGB-pixel hashes are disjoint between splits. Exact deduplication does not establish independence between nearby captures, the same physical objects or shared backgrounds. Repeated compositions do not create independent source photographs. Small or partial targets, retained context and approximate boxes can affect results. Annotations cover the assigned folder target; incidental classes are not exhaustively labelled. Human annotation checks and an independently photographed real-scene test set would be needed to assess real-world generalization.

The selected model is saved locally at the path above; weights and generated images are outside Git. A repository clone contains the code, source annotations, manifest, saved detections, training history, metrics and report, but requires data regeneration and training to produce weights. The report was rendered and every page visually checked. Saved metrics were independently recomputed from detections. All 10,000 generated image/label pairs passed validation, and the Colab GPU was stopped after artifact retrieval.

## References

- https://docs.ultralytics.com/models/yolov8/
- https://docs.ultralytics.com/datasets/detect/
- https://docs.ultralytics.com/modes/train/
- https://github.com/IDEA-Research/GroundingDINO
