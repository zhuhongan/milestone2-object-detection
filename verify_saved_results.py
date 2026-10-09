"""Recompute reported results from committed evidence without a GPU or raw photos."""
import csv
import hashlib
import json
from pathlib import Path
from evaluate_predictions import choose_threshold, evaluate

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="") as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    reports = ROOT / "reports"
    evidence = reports / "evidence"
    load = lambda path: json.loads(path.read_text())
    metrics = load(reports / "metrics.json")
    selection = load(reports / "selection.json")
    operating = load(reports / "test_operating_metrics.json")
    threshold_record = load(reports / "operating_threshold.json")
    protocol = load(reports / "training_protocol.json")
    audit = load(evidence / "dataset_audit.json")
    mapping = load(ROOT / "class_mapping.json")
    manifest_path = evidence / "generated_manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    history = read_csv(evidence / "training_history.csv")
    assert len(history) == 50
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == protocol["generated_manifest_sha256"]
    assert hashlib.sha256((ROOT / "annotations/source_annotations.json").read_bytes()).hexdigest() == protocol["source_annotations_sha256"]
    receipt = load(evidence / "best_epoch_receipt.json")
    assert receipt["epoch"] == selection["selected_epoch"]
    peak = max(float(row["metrics/mAP50-95(B)"]) for row in history)
    selected = next(row for row in history if int(float(row["epoch"])) == selection["selected_epoch"])
    assert float(selected["metrics/mAP50-95(B)"]) == peak == selection["validation_mAP50_95"]
    per_class = read_csv(reports / "per_class_metrics.csv")
    assert len(mapping) == 73
    for split in ("val", "test"):
        rows = [row for row in per_class if row["split"] == split]
        assert len(rows) == 73
        assert {row["object_id"] for row in rows} == {row["object_id"] for row in mapping}
        for field, key in (("AP50", "mAP50"), ("AP50_95", "mAP50_95")):
            assert abs(sum(float(row[field]) for row in rows) / 73 - metrics[split][key]) < 1e-10
    for split in ("train", "val", "test"):
        rows = [row for row in manifest if row["split"] == split]
        assert len(rows) == audit["counts"][split]["images"]
        assert sum(len(row["targets"]) for row in rows) == audit["counts"][split]["target_instances"]
        assert {target["object_id"] for row in rows for target in row["targets"]} == {row["object_id"] for row in mapping}
    val_predictions = read_csv(evidence / "val_predictions_candidates.csv")
    threshold, grid = choose_threshold(val_predictions, [row for row in manifest if row["split"] == "val"])
    assert threshold == threshold_record["confidence"] == operating["confidence"]
    assert grid == threshold_record["validation_grid"]
    test_predictions = read_csv(evidence / "test_predictions_candidates.csv")
    assert evaluate(test_predictions, [row for row in manifest if row["split"] == "test"], threshold) == operating
    fixed = [row for row in test_predictions if float(row["confidence"]) >= threshold]
    assert fixed == read_csv(evidence / "test_detections.csv")
    session = load(evidence / "session_receipt.json")
    assert session["gpu_stopped"] and session["exports_verified"]
    print(json.dumps({"verified": True, "epochs": 50, "selected_epoch": selection["selected_epoch"],
                      "composites": len(manifest), "threshold": threshold,
                      "test_mAP50": metrics["test"]["mAP50"],
                      "test_mAP50_95": metrics["test"]["mAP50_95"],
                      "test_all_objects": operating["all_objects_accuracy"],
                      "gpu_stopped": True}, indent=2))


if __name__ == "__main__":
    main()
