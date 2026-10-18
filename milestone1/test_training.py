#!/usr/bin/env python3
"""Meaningful training checks; opt in to actual ConvNeXt CPU/GPU integration.

python -m unittest test_training -v
RUN_TRAINING_SMOKE=1 python -m unittest test_training -v
RUN_TRAINING_SMOKE=1 SMOKE_DEVICE=cuda python -m unittest test_training -v
The synthetic results are discarded and must never be used in the report.
"""
import argparse
import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import train


def refresh_fixture_audit(prepared):
    """Sign synthetic fixtures; real runs use prepare_dataset.py's audit."""
    train.atomic_json(prepared / "audit.json", {
        "status": "success", "seed": 42, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "class_mapping_sha256": train.sha256(prepared / "class_mapping.json"),
        "manifest_sha256": {f"{split}.csv": train.sha256(prepared / f"{split}.csv") for split in ("train", "val", "test")},
    })


def fixture(directory):
    raw = directory / "raw"
    prepared = directory / "prepared"
    raw.mkdir()
    prepared.mkdir()
    classes = [{"folder": f"object_{i}", "object_id": f"{7 + i:03d}", "object_name": f"Test object {i}", "class_index": i} for i in range(2)]
    manifests = {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(42)
    for item in classes:
        folder = raw / item["folder"]
        folder.mkdir()
        for number in range(10):
            # Different resolutions and valid JPG/JPEG, including grayscale.
            width, height = 48 + number, 64 + 2 * number
            pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            if number == 0:
                image = image.convert("L")
            image_path = folder / f"{number}.{'jpg' if number % 2 else 'jpeg'}"
            image.save(image_path)
            with Image.open(image_path) as source:
                pixel_hash = hashlib.sha256(source.convert("RGB").tobytes()).hexdigest()
            split = "train" if number < 8 else ("val" if number == 8 else "test")
            manifests[split].append(dict(item, path=image_path.relative_to(raw).as_posix(),
                                         sha256=train.sha256(image_path), pixel_sha256=pixel_hash))
    for split, rows in manifests.items():
        train.write_csv(prepared / f"{split}.csv", rows, list(rows[0]))
    train.atomic_json(prepared / "class_mapping.json", classes)
    refresh_fixture_audit(prepared)
    return raw, prepared, classes


class TrainingChecks(unittest.TestCase):
    def test_frozen_training_updates_only_linear_head_and_disables_dropout(self):
        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.9))
                self.classifier = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(4, 2))

            def forward(self, x):
                return self.classifier(self.features(x))

        train.seed_everything(42)
        model = SmallModel()
        original_features = model.features[0].weight.detach().clone()
        original_head = model.classifier[2].weight.detach().clone()
        args = argparse.Namespace(frozen_lr=0.01, weight_decay=0.01)
        optimizer = train.configure_stage(model, "frozen", args)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]), torch.arange(4)), batch_size=4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        train.run_epoch(model, loader, torch.device("cpu"), 2, "frozen", 1, optimizer=optimizer, scaler=scaler)
        self.assertFalse(model.features[1].training)
        self.assertTrue(model.classifier[2].training)
        self.assertTrue(torch.equal(original_features, model.features[0].weight))
        self.assertFalse(torch.equal(original_head, model.classifier[2].weight))

    def test_macro_f1_includes_unpredicted_classes(self):
        result = train.metrics_from_confusion(np.array([[3, 0], [1, 0]]), 1.2)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], 3 / 7)
        self.assertEqual(result["per_class"][1]["f1"], 0)
        self.assertTrue(train.better({"macro_f1": .5, "loss": .4}, {"macro_f1": .5, "loss": .5}))
        self.assertFalse(train.better({"macro_f1": .49, "loss": .1}, {"macro_f1": .5, "loss": 1.0}))

    def test_zero_optimizer_updates_cannot_be_reported_as_training(self):
        class OverflowScaler:
            current_scale = 1024.0

            def get_scale(self):
                return self.current_scale

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def is_enabled(self):
                return True

            def step(self, optimizer):
                pass  # Simulate GradScaler's overflow path: no optimizer call.

            def update(self):
                self.current_scale /= 2

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]), torch.arange(4)), batch_size=4)
        with self.assertRaisesRegex(RuntimeError, "No optimizer updates"):
            train.run_epoch(model, loader, torch.device("cpu"), 2, "finetuned", 1, optimizer=optimizer, scaler=OverflowScaler())

    def test_manifest_rejects_duplicates_and_changed_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw, prepared, _ = fixture(directory)
            # Match CLI normalization, including macOS /var -> /private/var aliases.
            args = argparse.Namespace(raw_root=raw.resolve(), prepared_dir=prepared.resolve(), seed=42)
            _, manifests, _ = train.read_inputs(args)
            rows = manifests["test"]
            rows[0]["pixel_sha256"] = manifests["train"][0]["pixel_sha256"]
            train.write_csv(prepared / "test.csv", rows, list(rows[0]))
            with self.assertRaisesRegex(ValueError, "test.csv changed after"):
                train.read_inputs(args)
            refresh_fixture_audit(prepared)
            with self.assertRaisesRegex(ValueError, "Duplicate pixel_sha256"):
                train.read_inputs(args)
            rows[0]["pixel_sha256"] = "different_unique_pixel_hash"
            train.write_csv(prepared / "test.csv", rows, list(rows[0]))
            refresh_fixture_audit(prepared)
            image_path = raw / manifests["train"][0]["path"]
            image_path.write_bytes(image_path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Raw image changed"):
                train.read_inputs(args)

    def test_failed_audit_blocks_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw, prepared, _ = fixture(Path(temporary))
            train.atomic_json(prepared / "audit.json", {"status": "failed"})
            with self.assertRaisesRegex(ValueError, "audit is not successful"):
                train.read_inputs(argparse.Namespace(raw_root=raw, prepared_dir=prepared, seed=42))

    def test_output_cannot_be_inside_locked_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw, prepared, _ = fixture(Path(temporary))
            with self.assertRaises(SystemExit):
                train.parse_args(["--raw-root", str(raw), "--prepared-dir", str(prepared), "--output", str(raw / "run")])

    @unittest.skipUnless(os.environ.get("RUN_TRAINING_SMOKE") == "1", "set RUN_TRAINING_SMOKE=1 for actual ConvNeXt integration")
    def test_actual_convnext_two_stages_and_completed_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="convnext-smoke-") as temporary:
            directory = Path(temporary)
            raw, prepared, classes = fixture(directory)
            output = directory / "run"
            original = {path.relative_to(raw).as_posix(): train.sha256(path) for path in raw.rglob("*") if path.is_file()}
            argv = ["--raw-root", str(raw), "--prepared-dir", str(prepared), "--output", str(output),
                    "--weights", "none", "--device", os.environ.get("SMOKE_DEVICE", "cpu"),
                    "--workers", "0", "--batch-size", "16", "--frozen-epochs", "1", "--finetune-epochs", "1"]
            train.main(argv)
            selection = json.loads((output / "selection.json").read_text())
            self.assertFalse(selection["test_used_for_selection"])
            self.assertFalse(selection["pretrained"])
            comparisons = json.loads((output / "comparison.json").read_text())
            self.assertEqual(len(comparisons), 2)
            self.assertEqual(sum(row["selected_from_validation"] for row in comparisons), 1)
            self.assertTrue((output / "learning_curves.png").exists())
            expected_ids = {item["object_id"] for item in classes}
            for stage in train.STAGES:
                history = json.loads((output / stage / "history.json").read_text())
                self.assertEqual(history[0]["optimizer_updates"], 1)
                self.assertEqual(history[0]["skipped_optimizer_steps"], 0)
                with (output / stage / "test_predictions.csv").open(newline="") as stream:
                    predictions = list(csv.DictReader(stream))
                self.assertEqual({row["true_object_id"] for row in predictions}, expected_ids)
                self.assertEqual(len(predictions), 2)
                self.assertTrue((output / stage / "last.pt").exists())
            frozen = train.load_checkpoint(output / "frozen" / "last.pt")
            fine = train.load_checkpoint(output / "finetuned" / "last.pt")
            self.assertFalse(torch.equal(frozen["model"]["features.0.0.weight"], fine["model"]["features.0.0.weight"]))
            before_resume = {path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
            train.main(argv + ["--resume"])
            self.assertEqual(before_resume, {path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()})
            # Simulate a disconnect after held-out evaluation but before final
            # completion: checkpoint resume must not train or test a second time.
            test_mtimes = {stage: (output / stage / "test_metrics.json").stat().st_mtime_ns for stage in train.STAGES}
            (output / "completed.json").unlink()
            with patch.object(train, "run_epoch", side_effect=AssertionError("Completed epochs/test evaluation were repeated")):
                train.main(argv + ["--resume"])
            self.assertEqual(test_mtimes, {stage: (output / stage / "test_metrics.json").stat().st_mtime_ns for stage in train.STAGES})
            self.assertEqual(original, {path.relative_to(raw).as_posix(): train.sha256(path) for path in raw.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
