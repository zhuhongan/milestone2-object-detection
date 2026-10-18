# Milestone 2 Object Detection with YOLOv8

Milestone 1 classification code is in **[milestone1/](milestone1/README.md)**, including the four-model 20-epoch comparison, preprocessing, fixed seed-42 splits, Colab notebooks, report generation and the live ConvNeXt-Tiny demo. The object-detection pipeline below remains at the repository root.

Train a 73-class YOLOv8s detector using synthetic multi-object images from the finalized shared single-object dataset. Predictions export final Object IDs, names, confidence scores and pixel bounding boxes. The shared raw dataset remains read-only.

## Dataset and annotations

- Final dataset: https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS
- Final Object IDs: https://docs.google.com/spreadsheets/d/1OCBWlTK8JSzrgUxON3thJWAKRKU2Pi-V9Ssn2BbDxds/edit#gid=0
- `class_mapping.json` preserves all 73 final IDs and the corresponding zero-based YOLO indices.

All 7,343 valid source photos have annotation records in `annotations/all_photos_seed42/`. The 28 exact duplicate copies inherit canonical annotations and are excluded before splitting. Grounding DINO proposed boxes for 7,315 unique photos; the assistant reviewed 151 contact sheets and recorded 1,379 overrides. There is no independent human annotation verification.

The positive pool contains 6,933 unique photos and 7,080 target boxes. Another 382 unique photos were marked target-absent during assistant review; they remain intact with recorded reasons. Background filenames are not automatic exclusion rules. These annotations cover the assigned folder target; incidental classes are not exhaustively labelled, and empty source labels must not be treated as fully annotated background negatives.

## Mixed layouts and source separation

The dataset contains 10,000 images: **8,000 train / 1,000 validation / 1,000 test**, seed 42. Every split uses **40% 2×2, 40% 3×3 and 20% random placement**. Training has 3,200 / 3,200 / 1,600 images; validation and test each have 400 / 400 / 200. All 73 Object IDs appear in every split/layout.

Grid cells contain full EXIF-oriented RGB positive source photos. Resizing preserves aspect ratio; padding, 2–10% inner margins and varying cell position provide variation. Every cell has an annotated target, with separate boxes for repeated instances. There are no background-only composites or empty cells. Random images combine 2–4 distinct classes using 5–15% contextual crops, random scale and non-overlapping placement. Training photos receive horizontal flips and mild brightness/contrast adjustments; validation and test composites are frozen.

Every eligible positive source is used. Source photos retain the established seed-42 split: 5,549 train / 696 validation / 688 test. Source paths, byte hashes and RGB-pixel hashes are disjoint between splits. Full generation, validation and environment records are in [the dataset audit](annotations/mixed_layouts_seed42/DATASET_AUDIT.md).

```bash
python generate_mixed_dataset.py --raw-root /path/to/final-local/images \
  --output data/mixed_layouts_seed42 --counts 8000 1000 1000 --seed 42
python validate_dataset.py data/mixed_layouts_seed42
python -m unittest discover -s tests -v
```

## Colab training protocol

The COCO-pretrained YOLOv8s model completed **50 epochs** on **NVIDIA A100-SXM4-40GB** through Colab CLI. Input size is 640, batch 32, seed 42, with no early stopping. AdamW uses initial LR 0.001, cosine final factor 0.01, weight decay 0.0005 and three warm-up epochs. Training-only augmentations include mosaic (closed for the last ten epochs), ±10° rotation, translation 0.1, scale 0.3, mild HSV changes and horizontal flip 0.5. RGB tensors are scaled to 0–1.

Validation mAP50–95 selected epoch **46**; test data did not select the checkpoint. The operating confidence **0.55** maximizes validation micro-F1 at matching IoU 0.50, then stays fixed for all test layouts. NMS IoU is 0.7. Exact parameters and package versions are archived with the report.

```bash
python run_colab.py --session milestone2-yolov8s-mixed-YOUR_UNIQUE_NAME \
  --gpu A100 --dataset data/mixed_layouts_seed42 \
  --output artifacts/yolov8s_mixed_layouts_seed42
```

The launcher validates the dataset, uploads checksum-verified parts with bounded transport retries, trains and evaluates, retrieves verified artifacts and releases the GPU. A failed export preserves the runtime for recovery using the same session/config/output and `--recover`. Session configuration files are private and excluded from Git.

## Measured results

On all 1,000 test images, **mAP50 is 96.10% and mAP50–95 is 84.93%**. Fixed-threshold micro precision is 96.68%, recall 94.12% and F1 0.9538: 5,535 TP, 190 FP and 346 missed targets. Every annotated object is correctly detected with no false positives in 690/1,000 test images (69.00%).

| Test layout | Images | mAP50 | mAP50–95 | Micro F1 | All objects correct |
|---|---:|---:|---:|---:|---:|
| 2×2 | 400 | 96.04% | 84.68% | 0.9503 | 74.00% |
| 3×3 | 400 | 95.63% | 84.29% | 0.9521 | 54.25% |
| Random | 200 | 98.74% | 91.89% | 0.9736 | 88.50% |

Layout scores use the same checkpoint and global validation-selected confidence. Overall AP is evaluated across all test images, rather than averaged from layout scores. Library precision/recall are class-mean curve summaries; operational micro metrics use the fixed confidence and class-aware IoU matching.

- [Word report with plots and detection examples](reports/mixed_layouts_seed42/milestone2_report.docx)
- [Saved results and independently verified evidence](reports/mixed_layouts_seed42/)
- [Plain 2×2 sample](samples/mixed_layouts_seed42/2x2.jpg)
- [Plain 3×3 sample](samples/mixed_layouts_seed42/3x3.jpg)
- [Ground truth, detections and final IDs for the samples](samples/mixed_layouts_seed42/)

Two test paths per layout were sampled with seed 42 independently of predictions. Plain samples are actual dataset images suitable for the requested demo preparation; ground-truth and model-output versions are saved separately.

## Saved model and inference

The selected checkpoint is saved locally at `artifacts/yolov8s_mixed_layouts_seed42/experiment/runs/yolov8s_seed42/weights/best.pt`.
SHA-256: `aea56ef314265da291ab40244d3a71ed0022f837c8ad11b8c4ee1d012c0bf71d`.
Weights and generated datasets stay outside Git. A repository clone contains code, annotations, manifests, saved predictions, metrics, histories, sample images and the report; it requires data regeneration and training to create model weights.

```bash
python predict.py --weights artifacts/yolov8s_mixed_layouts_seed42/experiment/runs/yolov8s_seed42/weights/best.pt \
  --source /path/to/multi_object_image.jpg --device cpu \
  --threshold-file reports/mixed_layouts_seed42/operating_threshold.json \
  --output predictions_mixed
python verify_saved_results.py --reports reports/mixed_layouts_seed42
```

The annotated image and `detections.json` contain final Object IDs, names, confidences and `(x1,y1,x2,y2)` locations in original-image pixels. Confidence is not a calibrated correctness probability.

## Reproduce the report

Use Python 3.12 and `generation-requirements.txt` for synthesis. Use a separate compatible PyTorch environment with `requirements.txt` for training/inference, and `report-requirements.txt` for figures/Word. Matching seeds alone do not guarantee identical bytes or GPU results across software/hardware stacks.

```bash
python analyze_results.py --experiment artifacts/yolov8s_mixed_layouts_seed42/experiment \
  --dataset data/mixed_layouts_seed42 --output reports/mixed_layouts_seed42
python render_word_report.py --data reports/mixed_layouts_seed42/report_data.json \
  --output reports/mixed_layouts_seed42/milestone2_report.docx
```

All 10,000 image/label pairs passed validation; metrics were recomputed from saved predictions. Every report page was rendered and visually checked. Exports were verified by SHA-256 and the dedicated GPU was stopped.

## Interpretation

These are synthetic benchmark results. Repeated compositions do not create independent photographs, and exact deduplication does not establish independence between nearby captures or shared physical objects/backgrounds. Small or partial targets, approximate annotations and unlabelled incidental classes can affect scores. An independently annotated real-scene test set and human annotation checks are needed before claiming real-world accuracy. Testing both grid sizes supports demo preparation without guaranteeing success on every unseen image.

## References

- https://docs.ultralytics.com/models/yolov8/
- https://docs.ultralytics.com/datasets/detect/
- https://github.com/IDEA-Research/GroundingDINO
