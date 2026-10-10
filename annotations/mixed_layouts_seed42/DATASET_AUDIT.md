# Mixed layout dataset audit

The finalized source photos and final Object IDs are unchanged. Source annotations cover the assigned folder object and were generated with Grounding DINO proposals, then reviewed and corrected by the assistant. No independent human verification is claimed.

Every one of the 6,933 eligible positive source photos is used. The exact-deduplicated seed-42 source split is preserved: 5,549 train, 696 validation and 688 test photos. Source paths, byte hashes and RGB-pixel hashes are disjoint between splits. The 382 unique photos marked target-absent during annotation review remain intact and are excluded from the positive synthesis pool. Filename background hints are not automatic exclusion rules.

There are 10,000 composites, with an 80/10/10 split and seed 42. Each split contains 40% 2x2 grids, 40% 3x3 grids and 20% random-placement images. Training has 3,200 / 3,200 / 1,600 images; validation and test each have 400 / 400 / 200. All 73 final IDs appear in every split/layout.

Grid cells retain full EXIF-oriented RGB source photographs. Photos are resized with aspect ratio preserved, padded, and placed with 2–10% inner margins and varying position inside the assigned cell. Every cell contains one annotated positive source; repeated target instances retain separate boxes. Grids have four or nine cells, so they can contain more than four or nine targets. There are no background-only composites or empty cells.

Random images select 2–4 distinct classes, use 5–15% contextual crops, and vary scale and placement without overlap. Training source augmentations are horizontal flip and mild brightness/contrast changes. Validation and test composites are frozen without these source augmentations. All coordinate transforms are recorded in the manifest.

All 10,000 generated image/label pairs passed checks for file hashes, normalized coordinates, final IDs, source separation, full positive source coverage, non-overlapping patches, occupied grid cells, target/cell containment and class coverage by layout. Fifteen local pipeline tests passed, plus three CLI recovery tests. Generation environment and manifest/source checksums are recorded alongside this audit.

Repeated compositions do not create independent photographs. Full grid photos can contain incidental classes that were not exhaustively annotated. Near captures, shared objects/backgrounds, small or partial targets and approximate annotations remain limitations. Results will describe this synthetic dataset and require a separately annotated real-scene test set to establish real-world accuracy.
