# Expanded source annotation audit

All 7,343 photographs in the finalized local dataset were inventoried without changing their bytes. Final Object IDs OBJ001–OBJ073 and names are preserved from the supplied sheet. The established Milestone 1 manifests provide exact duplicate decisions and source splits (seed 42).

| Quantity | Count |
|---|---:|
| Raw photo annotation records / YOLO label files | 7,343 |
| Unique source photos inspected | 7,315 |
| Exact duplicate copies, annotated through canonical aliases | 28 |
| Unique photos with visible assigned targets | 6,933 |
| Unique photos with no identifiable assigned target | 382 |
| Visible assigned-target bounding boxes in the unique positive pool | 7,080 |
| Eligible positive train / validation / test sources | 5,549 / 696 / 688 |
| Initial contact sheets inspected | 151 |
| Explicit per-photo review overrides | 1,379 |

Grounding DINO generated proposals for every unique source on an NVIDIA A100 through Colab CLI. The downloaded exports were checksum verified and the GPU was stopped. The exact model revision and installed environment are archived alongside this audit. Existing 365 source annotations were retained as initial drafts; further visual review could correct them.

The assistant inspected contact sheets, removed proposals on unrelated objects, corrected missed or misplaced boxes, and added visible repeated instances. Difficult examples were enlarged with coordinate grids. Manual box changes received an additional contact-sheet check, and generated examples were inspected. These are model-assisted, assistant-reviewed annotations. They are not independently human-verified ground truth and should receive human spot checks before final submission.

Coordinates use normalized xyxy on the EXIF-oriented primary RGB frame (frame 0 for MPO files). YOLO labels convert to zero-based class index and normalized centre-x, centre-y, width, height. Partial visible objects are labelled when identifiable. Separate instances receive separate boxes. Background filename hints are not exclusion rules. A target-absent source is excluded from object synthesis; its original photograph remains intact. Duplicate copies inherit canonical annotations and splits, and are excluded from synthesis to avoid redundant sources.

**Scope:** these annotations cover visible instances of the assigned folder Object ID. Incidental other classes are not exhaustively annotated. An empty source label is not an annotation that every target class is absent, and these files must not be treated as a fully labelled direct-training dataset of background negatives. Context retained inside pasted crops can include incidental objects.

Synthesis cycles through seeded shuffled class pools and uses every eligible positive source at least once. Source splits are assigned before synthesis. The expanded frozen dataset has 8,000 train, 1,000 validation and 1,000 test composites (10,000 total), with all 73 classes in each split. Each composite uses 2–4 distinct classes and retains repeated instances. Rectangular contextual crops are resized and placed without overlap on synthetic backgrounds. Training-only source augmentations are horizontal flip and mild brightness/contrast changes.

`expanded_dataset_audit.json` and `expanded_dataset_validation.json` archive the generated dataset checks. Twelve pipeline tests passed, covering coordinate transforms, matching, seed reproducibility, complete source coverage, and rejecting an insufficient coverage budget. Raw-image checksums were reverified when finalizing annotations and generating composites.

The earlier 365-source / 1,460-composite model, Word report and metrics remain archived as the baseline. The expanded experiment is now completed; its measured results and updated Word report are archived in `reports/all_photos_seed42/`. More distinct sources increase available image variety; they do not guarantee improved detection accuracy. Near-duplicate captures, shared backgrounds and the gap between synthetic composites and real multi-object scenes remain limitations.
