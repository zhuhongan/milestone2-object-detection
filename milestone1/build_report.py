#!/usr/bin/env python3
"""Build a Milestone 1 Markdown report exclusively from a completed experiment.

Only standard-library modules are used. Downloaded artifacts, including prepared/
and runs/convnext_tiny_seed42/, must be beneath --experiment. All report file links
are relative to --output and never use the Colab paths saved in config.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote


DATASET_URL = "https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS"
IDS_URL = "https://docs.google.com/spreadsheets/d/1OCBWlTK8JSzrgUxON3thJWAKRKU2Pi-V9Ssn2BbDxds/edit#gid=0"
STAGES = ("frozen", "finetuned")
LABELS = {"frozen": "Frozen backbone + trained linear classifier", "finetuned": "Full-network fine-tuning"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell(value) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")


def table(headers, rows) -> str:
    result = ["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(result)


def number(value, name: str, upper: float | None = None) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"Missing/non-numeric {name}")
    value = float(value)
    require(math.isfinite(value) and value >= 0 and (upper is None or value <= upper), f"Invalid {name}")
    return value


def describe_transforms(config: dict) -> str:
    """Translate this experiment's saved recipe without inventing new settings."""
    training = re.fullmatch(
        r"EXIF transpose, RGB, RandomResizedCrop(\d+) scale([\d.]+)-([\d.]+) "
        r"ratio(\d+)/(\d+)-(\d+)/(\d+) bilinear antialias; horizontal flip p([\d.]+); "
        r"brightness/contrast/saturation jitter([\d.]+), hue([\d.]+); ToTensor; ImageNet normalization",
        config["transforms"]["train"],
    )
    evaluation = re.fullmatch(
        r"EXIF transpose, RGB, resize short side(\d+) bilinear antialias, center crop(\d+), ToTensor, ImageNet mean/std",
        config["transforms"]["eval"],
    )
    require(training is not None and evaluation is not None, "The saved transform recipe is not recognized; update the report description to match it.")
    size, area_low, area_high, ratio_ln, ratio_ld, ratio_hn, ratio_hd, flip, jitter, hue = training.groups()
    low_ratio, high_ratio = int(ratio_ln) / int(ratio_ld), int(ratio_hn) / int(ratio_hd)
    short_side, crop = evaluation.groups()
    return (
        f"For training, a random crop targets **{100 * float(area_low):g}–{100 * float(area_high):g}% of the image area** "
        f"with a width-to-height ratio of {ratio_ln}/{ratio_ld}–{ratio_hn}/{ratio_hd} "
        f"(approximately {low_ratio:.2f}–{high_ratio:.3f}), then is resized to **{size} × {size} pixels**. "
        f"Horizontal flipping occurs with **{100 * float(flip):g}% probability**. Color jitter varies brightness, contrast, "
        f"and saturation by up to {100 * float(jitter):g}% and hue by up to {float(hue):g}. "
        f"Validation and test images are resized so the shorter side is **{short_side} pixels**, "
        f"then centrally cropped to **{crop} × {crop} pixels**. Resizing uses bilinear interpolation with antialiasing. "
        "RGB values are scaled to [0, 1] and normalized with fixed ImageNet mean (0.485, 0.456, 0.406) "
        "and standard deviation (0.229, 0.224, 0.225). Random augmentation applies only to training."
    )


def convergence_analysis(config: dict, histories: dict, results: dict, selected: str) -> str:
    frozen_first, frozen_last = histories["frozen"][0], histories["frozen"][-1]
    chosen_history = histories[selected]
    chosen_last = chosen_history[-1]
    chosen_best = results[selected]
    final_epoch = chosen_last["epoch"]
    best_epoch = chosen_best["best_epoch"]
    gap = 100 * (chosen_last["train_accuracy"] - chosen_last["val_accuracy"])
    val_test_gap = 100 * (chosen_best["validation_macro_f1"] - chosen_best["test_macro_f1"])
    paragraphs = [
        f"The frozen classifier's validation macro-F1 rose from **{100 * frozen_first['val_macro_f1']:.2f}%** "
        f"in epoch {frozen_first['epoch']} to its best **{100 * results['frozen']['validation_macro_f1']:.2f}%** "
        f"in epoch {results['frozen']['best_epoch']}. By its final epoch {frozen_last['epoch']}, "
        f"recorded training accuracy was {100 * frozen_last['train_accuracy']:.2f}% versus "
        f"{100 * frozen_last['val_accuracy']:.2f}% validation accuracy.",
        f"For the selected {LABELS[selected].lower()} approach, the final completed epoch was **{final_epoch}**. "
        f"Its recorded training accuracy was **{100 * chosen_last['train_accuracy']:.2f}%**, versus "
        f"**{100 * chosen_last['val_accuracy']:.2f}%** on validation (a training-minus-validation gap of "
        f"**{gap:+.2f} percentage points**); training and validation losses were "
        f"**{chosen_last['train_loss']:.4f}** and **{chosen_last['val_loss']:.4f}**, respectively. "
        "Training metrics are measured during augmented training, whereas validation uses fixed evaluation crops, "
        "so this comparison describes the observed behavior rather than a controlled estimate of overfitting.",
    ]
    trailing = [row for row in chosen_history if row["epoch"] > best_epoch]
    maximum = config["frozen_epochs"] if selected == "frozen" else config["finetune_epochs"]
    best_score = (chosen_best["validation_macro_f1"], -chosen_best["validation_loss"])
    no_later_improvement = all((row["val_macro_f1"], -row["val_loss"]) <= best_score for row in trailing)
    if final_epoch < maximum and len(trailing) >= config["patience"] and no_later_improvement:
        paragraphs.append(
            f"The selected checkpoint remained **epoch {best_epoch}**. The next **{len(trailing)} epochs** did not improve "
            f"the validation selection criterion, reaching the configured patience of **{config['patience']}** and "
            f"stopping at epoch **{final_epoch}** before the {maximum}-epoch limit. "
            f"Final validation macro-F1 was **{100 * chosen_last['val_macro_f1']:.2f}%**, compared with "
            f"**{100 * chosen_best['validation_macro_f1']:.2f}%** at the chosen checkpoint. "
            "Additional fitting therefore did not improve the selected validation score."
        )
    else:
        paragraphs.append(
            f"The final epoch and chosen checkpoint serve different roles: the run completed epoch **{final_epoch}**, "
            f"while **epoch {best_epoch}** had the best validation selection score and supplies the reported test results."
        )
    paragraphs.append(
        f"For the selected checkpoint, validation-minus-test macro-F1 is **{val_test_gap:+.2f} percentage points**. "
        "This held-out difference and the improvements between approaches are descriptive results from one split; "
        "their reliability should be assessed with repeated splits or additional independent images."
    )
    return "\n\n".join(paragraphs)


def render_report(experiment: Path, output: Path) -> str:
    experiment, output = experiment.resolve(), output.resolve()
    prepared = experiment / "prepared"
    run = experiment / "runs" / "convnext_tiny_seed42"
    completed = read_json(run / "completed.json")
    require(completed.get("pretrained") is True, "Report requires a completed pretrained experiment; smoke/unpretrained runs are rejected.")
    config = read_json(run / "config.json")
    selection = read_json(run / "selection.json")
    audit = read_json(prepared / "audit.json")
    classes = read_json(run / "class_mapping.json")
    comparison = read_json(run / "comparison.json")
    environment = read_json(run / "environment.json")
    require(audit.get("status") == "success", "A successful preprocessing audit is required.")
    require(all(audit.get("leakage_checks", {}).get(key) is True for key in ("paths_disjoint", "raw_sha256_disjoint", "rgb_pixel_sha256_disjoint")), "All preprocessing leakage checks must pass.")
    require(config.get("architecture") == "convnext_tiny" and config.get("weights") == "IMAGENET1K_V1", "Expected ConvNeXt-Tiny ImageNet transfer learning.")
    require(config.get("seed") == audit.get("seed") == 42, "Expected seed 42 for preprocessing and training.")
    require(audit.get("ratios") == {"train": 0.8, "val": 0.1, "test": 0.1}, "Expected target split 80/10/10.")
    require(selection.get("pretrained") is True and selection.get("test_used_for_selection") is False, "Validation-only model selection must be recorded.")
    require(isinstance(classes, list) and classes, "Missing official Object ID mapping.")
    names = {row["object_id"]: row["object_name"] for row in classes}
    require(len(names) == len(classes) and all(isinstance(key, str) for key in names), "Official Object IDs must be unique strings.")
    require(classes == read_json(prepared / "class_mapping.json"), "Run and preprocessing Object ID mappings differ.")
    require(sha256(prepared / "class_mapping.json") == audit.get("class_mapping_sha256"), "Class mapping differs from its preprocessing audit.")
    for split in ("train", "val", "test"):
        require(sha256(prepared / f"{split}.csv") == audit.get("manifest_sha256", {}).get(f"{split}.csv"), f"{split} manifest differs from its audit.")
        require(len(read_csv(prepared / f"{split}.csv")) == audit["split_counts"][split], f"{split} sample count differs from its audit.")
    require(sum(audit["split_counts"].values()) == audit["retained_images"], "Split totals differ from the retained image count.")
    for filename, digest in config["manifest_sha256"].items():
        require(sha256(prepared / filename) == digest, f"Training used different {filename}; refusing to mix artifacts.")
    require(len(comparison) == len(STAGES) and {row["approach"] for row in comparison} == set(STAGES), "Both predefined approaches must have completed results.")
    results = {row["approach"]: row for row in comparison}
    histories = {}
    for stage, result in results.items():
        require(result.get("pretrained") is True, f"{stage} is not a pretrained experiment.")
        histories[stage] = read_json(run / stage / "history.json")
        require(histories[stage], f"{stage} has no completed epoch history.")
        require([row["epoch"] for row in histories[stage]] == list(range(1, len(histories[stage]) + 1)), f"{stage} epoch history is incomplete or out of order.")
        for epoch in histories[stage]:
            for split in ("train", "val"):
                for metric in ("accuracy", "macro_f1", "loss"):
                    number(epoch[f"{split}_{metric}"], f"{stage} epoch {epoch['epoch']} {split} {metric}", None if metric == "loss" else 1.0)
        require(result["best_epoch"] in [row["epoch"] for row in histories[stage]], f"{stage} best epoch is absent from its history.")
        best_validation = read_json(run / stage / "best_validation.json")
        test_metrics = read_json(run / stage / "test_metrics.json")
        for metric in ("accuracy", "macro_f1", "loss"):
            upper = None if metric == "loss" else 1.0
            validation_value = number(result[f"validation_{metric}"], f"{stage} validation {metric}", upper)
            test_value = number(result[f"test_{metric}"], f"{stage} test {metric}", upper)
            require(math.isclose(validation_value, best_validation[metric], abs_tol=1e-10), f"{stage} validation summary disagrees with comparison.")
            require(math.isclose(test_value, test_metrics[metric], abs_tol=1e-10), f"{stage} test summary disagrees with comparison.")
        for split in ("val", "test"):
            marker = read_json(run / stage / f"{split}_evaluation_complete.json")
            require(marker["samples"] == audit["split_counts"][split], f"{stage} {split} evaluation sample count mismatch.")
    selected = max(STAGES, key=lambda stage: (results[stage]["validation_macro_f1"], -results[stage]["validation_loss"]))
    require(selection.get("selected_stage") == completed.get("selected_stage") == selected, "Recorded selected model disagrees with validation metrics.")
    require(all(row.get("selected_from_validation") is (stage == selected) for stage, row in results.items()), "Comparison selection flags disagree.")
    checkpoint = run / selected / "best.pt"
    require(checkpoint.is_file(), "Selected model checkpoint is missing.")
    require(selection.get("checkpoint") == completed.get("selected_checkpoint") == f"{selected}/best.pt", "Selected checkpoint paths disagree.")
    selected_hash = sha256(checkpoint)
    for split in ("val", "test"):
        require(read_json(run / selected / f"{split}_evaluation_complete.json")["checkpoint_sha256"] == selected_hash, "Selected checkpoint differs from the evaluated checkpoint.")

    per_class = read_csv(run / selected / "test_per_class.csv")
    supports = {row["object_id"]: int(row["support"]) for row in per_class}
    require(set(supports) == set(names) and min(supports.values()) > 0, "Test per-class support/mapping is incomplete.")
    require(sum(supports.values()) == audit["split_counts"]["test"], "Per-class test supports do not total the test split.")
    confusion = read_csv(run / selected / "test_confusion_matrix.csv")
    require(len(confusion) == len(names) and {row["true_object_id"] for row in confusion} == set(names), "Confusion matrix Object IDs are incomplete.")
    errors, correct, evaluated = [], 0, 0
    for row in confusion:
        actual = row["true_object_id"]
        counts = {predicted: int(row[f"predicted_{predicted}"]) for predicted in names}
        require(all(value >= 0 for value in counts.values()) and sum(counts.values()) == supports[actual], "Confusion matrix counts disagree with class support.")
        evaluated += sum(counts.values())
        correct += counts[actual]
        errors.extend((count, actual, predicted) for predicted, count in counts.items() if count and predicted != actual)
    require(math.isclose(correct / evaluated, results[selected]["test_accuracy"], abs_tol=1e-10), "Confusion matrix accuracy disagrees with test metrics.")
    errors.sort(key=lambda item: (-item[0], item[1], item[2]))

    def link(label: str, target: Path) -> str:
        require(target.is_file(), f"Linked artifact is missing: {target}")
        relative = Path(os.path.relpath(target, output.parent)).as_posix()
        return f"[{label}]({quote(relative, safe='/._-')})"

    counts = audit["split_counts"]
    exclusions = audit.get("exclusion_counts", {})
    duplicate_count = sum(exclusions.get(key, 0) for key in ("exact_file_duplicate", "exact_rgb_pixel_duplicate"))
    metadata_count = exclusions.get("unsupported_nonimage_file", 0)
    valid_count = audit["valid_images_before_deduplication"]
    selected_result = results[selected]
    validation_delta = 100 * (results["finetuned"]["validation_macro_f1"] - results["frozen"]["validation_macro_f1"])
    test_delta = 100 * (results["finetuned"]["test_macro_f1"] - results["frozen"]["test_macro_f1"])
    report = [
        "# Milestone 1: ConvNeXt-Tiny object classification",
        "## Dataset and task",
        f"The task is supervised classification of images into **{len(classes)} object classes** using the locked [final merged dataset]({DATASET_URL}). Labels follow the [final Object ID Sheet]({IDS_URL}); official IDs remain strings and are separate from the model's contiguous class indices. The shared dataset was not changed.",
        f"The audit inspected **{audit['scanned_files']:,} files**: **{valid_count:,} valid images** and **{metadata_count:,} non-image metadata files**. After excluding **{duplicate_count:,} confirmed exact duplicates** from split manifests, **{audit['retained_images']:,} images** remained. No per-folder cap was applied.",
        "## Reproducible preprocessing",
        f"Images were verified and decoded, EXIF orientation was applied in memory, and pixels were converted to RGB. JPEG/MPO containers use **primary frame 0**, matching the training loader; auxiliary images such as HDR gain maps are not separate samples. The audit accepted **{audit['valid_multiframe_jpeg_mpo_primary_images']:,} multiframe JPEG/MPO primary photos**. Valid image dimensions and high resolution alone were not grounds for exclusion; resizing occurs online during training/evaluation.",
        "Before splitting, exact file SHA-256 and EXIF-oriented RGB pixel SHA-256 (including dimensions) were checked globally. A deterministic representative was retained for each same-label duplicate group; conflicting labels would stop preparation. Source files were neither deleted nor edited.",
        table(("Recorded exclusion reason", "Files"), [(key.replace("_", " "), f"{value:,}") for key, value in sorted(exclusions.items())]) if exclusions else "The audit recorded no exclusions.",
        f"The split is stratified by class with an **80/10/10 target and seed {audit['seed']}**: **{counts['train']:,} training / {counts['val']:,} validation / {counts['test']:,} test images**. Per-class largest-remainder integer rounding assigns every retained image (ties: training, validation, test), so overall percentages can differ slightly from their targets. Paths, raw hashes, and decoded pixel hashes are disjoint across splits.",
        describe_transforms(config),
        "## Models and experiment settings",
        "Both approaches use Torchvision **ConvNeXt-Tiny pretrained on ImageNet-1K**, with the final linear layer replaced for the dataset's classes. The first approach trains this classifier with the backbone frozen. The second starts from the best frozen checkpoint and fine-tunes the whole network. These are **two transfer-learning approaches using one CNN architecture**.",
        table(("Setting", "Frozen classifier", "Full-network fine-tuning"), [
            ("Maximum epochs", config["frozen_epochs"], config["finetune_epochs"]),
            ("Completed epochs", len(histories["frozen"]), len(histories["finetuned"])),
            ("AdamW learning rate", config["frozen_lr"], f"Backbone {config['backbone_lr']}; classifier {config['head_lr']}"),
            ("Weight decay", config["weight_decay"], config["weight_decay"]),
            ("Early-stopping patience", config["patience"], config["patience"]),
        ]),
        f"Both use batch size **{config['batch_size']}**, cosine learning-rate scheduling, gradient norm clipping at 1.0, and {cell(config['mixed_precision'])} mixed precision on **{cell(environment.get('gpu') or environment['device'])}**. The recorded environment uses PyTorch {cell(environment['torch'])} and Torchvision {cell(environment['torchvision'])}. Seeds cover initialization, shuffling, augmentation workers, and CUDA; deterministic algorithms are requested.",
        "## Results and model selection",
        "Checkpoint and approach selection used the highest **validation macro-F1**, with lower validation cross-entropy breaking ties. Selection was recorded before final held-out test evaluation; test metrics did not choose the model. Accuracy and macro-F1 below are percentages; cross-entropy loss is unitless.",
        table(("Approach", "Best epoch", "Val accuracy", "Val macro-F1", "Val loss", "Test accuracy", "Test macro-F1", "Test loss"), [
            (LABELS[stage] + (" — selected" if stage == selected else ""), results[stage]["best_epoch"],
             f"{100 * results[stage]['validation_accuracy']:.2f}%", f"{100 * results[stage]['validation_macro_f1']:.2f}%", f"{results[stage]['validation_loss']:.4f}",
             f"{100 * results[stage]['test_accuracy']:.2f}%", f"{100 * results[stage]['test_macro_f1']:.2f}%", f"{results[stage]['test_loss']:.4f}")
            for stage in STAGES
        ]),
        f"The selected approach is **{LABELS[selected]}**, using epoch **{selected_result['best_epoch']}**. Its validation macro-F1 is **{100 * selected_result['validation_macro_f1']:.2f}%**, and its held-out test macro-F1 is **{100 * selected_result['test_macro_f1']:.2f}%**. Fine-tuning minus the frozen approach changes validation macro-F1 by **{validation_delta:+.2f} percentage points** and test macro-F1 by **{test_delta:+.2f} percentage points**. Selection follows the validation ranking regardless of the test ranking.",
        "## Convergence and generalization",
        "!" + link("Training and validation learning curves for the frozen and fine-tuned approaches", run / "learning_curves.png"),
        convergence_analysis(config, histories, results, selected),
        "## Classification errors",
        f"For the selected model, **{evaluated - correct:,} of {evaluated:,} test images** were misclassified. The largest directed confusion counts are listed below with the official Object IDs and names; tied counts are ordered by ID.",
        table(("True Object ID and name", "Predicted Object ID and name", "Errors", "True-class test support"), [
            (f"{actual}: {names[actual]}", f"{predicted}: {names[predicted]}", count, supports[actual])
            for count, actual, predicted in errors[:8]
        ]) if errors else "The selected model's test confusion matrix has no off-diagonal errors.",
        "## Limitations and final artifacts",
        f"Each class has only **{min(supports.values())}–{max(supports.values())} test images**, so class-level estimates are sensitive to individual mistakes. Exact duplicate removal does not remove visually similar photographs of the same object or scene; an image-level split may retain that correlation. This experiment uses one split seed and one CNN architecture, so it does not establish performance across repeated splits, unseen capture settings, or alternative architectures. Reproducibility across different hardware and library versions is not guaranteed.",
        "Selected checkpoint: " + link(f"{selected}/best.pt", checkpoint) + ".",
        "Supporting artifacts: " + "; ".join([
            link("learning curves", run / "learning_curves.png"),
            link("full comparison", run / "comparison.csv"),
            link("per-class test metrics", run / selected / "test_per_class.csv"),
            link("test confusion matrix", run / selected / "test_confusion_matrix.csv"),
            link("preprocessing audit", prepared / "audit.json"),
            link("exclusion log", prepared / "exclusions.csv"),
            link("official Object ID mapping", run / "class_mapping.json"),
            link("experiment configuration", run / "config.json"),
        ]) + ".",
    ]
    return "\n\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".md":
        parser.error("--output must be a Markdown (.md) file")
    try:
        report = render_report(args.experiment, args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, prefix=".report-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(report)
        try:
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Report not generated: {error}\n")
    print(f"Saved verified experiment report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
