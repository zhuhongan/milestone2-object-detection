"""Run the Colab experiment and atomically package byte-verified export parts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path("/content/milestone1")
PART_BYTES = 32 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def package_archive(path, entries):
    # A killed packager leaves only .partial; a final name always has a closed tar.
    temporary = path.with_name(path.name + ".partial")
    with tarfile.open(temporary, "w:gz") as archive:
        for source, arcname in entries:
            archive.add(source, arcname=arcname)
    temporary.replace(path)


def split_archive(path, kind, experiment=None):
    parts_dir = path.with_name(path.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    parts, whole, total = [], hashlib.sha256(), 0
    with path.open("rb") as source:
        index = 0
        while block := source.read(PART_BYTES):
            part = parts_dir / f"part-{index:05d}"
            temporary = part.with_name(part.name + ".partial")
            temporary.write_bytes(block)
            temporary.replace(part)
            digest = hashlib.sha256(block).hexdigest()
            parts.append({"name": part.name, "remote_path": str(part),
                          "size_bytes": len(block), "sha256": digest})
            whole.update(block)
            total += len(block)
            index += 1
    manifest = {"schema_version": 1, "kind": kind, "complete": True,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "archive_name": path.name, "remote_archive": str(path),
                "size_bytes": total, "sha256": whole.hexdigest(),
                "part_bytes": PART_BYTES, "parts": parts}
    if experiment is not None:
        manifest["experiment"] = experiment
    manifest_path = path.with_name(path.name + ".manifest.json")
    # Readers see a manifest only after every referenced part is complete.
    atomic_json(manifest_path, manifest)
    print(f"Export ready: {manifest_path} ({len(parts)} parts, {total} bytes)", flush=True)
    return manifest


def experiment_summary():
    run_dir = ROOT / "runs/convnext_tiny_seed42"
    summary = {"completed": False}
    for name in ("completed", "selection", "comparison"):
        path = run_dir / f"{name}.json"
        if path.exists():
            summary[name] = json.loads(path.read_text())
    audit_path = ROOT / "prepared/audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        summary["preprocessing"] = {key: audit[key] for key in
            ("status", "seed", "split_ratios", "counts", "split_counts", "class_count") if key in audit}
    return summary


def run(arguments, log_name):
    with (ROOT / log_name).open("a") as log:
        process = subprocess.Popen([sys.executable, "-u", *arguments], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"{arguments[0]} failed: exit {process.returncode}; see {log_name}")


def main():
    ROOT.mkdir(exist_ok=True)
    os.chdir(ROOT)
    try:
        acquisition = json.loads((ROOT / "data/final/acquisition_manifest.json").read_text())
        if acquisition.get("complete") is not True:
            raise RuntimeError("A complete acquisition manifest is required before training")
        snapshot = ROOT.parent / "milestone1_raw.tar.gz"
        # Rebuild on each invocation: no existence-only reuse of interrupted archives.
        package_archive(snapshot, [(ROOT / "data/final", "final")])
        split_archive(snapshot, "raw")
        os.environ["RUN_TRAINING_SMOKE"] = "1"
        os.environ["SMOKE_DEVICE"] = "cuda"
        verification_path = ROOT / "verification_passed.json"
        verified_sources = {name: sha256(ROOT / name) for name in
            ("prepare_dataset.py", "download_dataset.py", "train.py",
             "test_prepare_dataset.py", "test_download_dataset.py", "test_training.py")}
        prior_verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
        if prior_verification.get("sources") != verified_sources:
            run(["-m", "unittest", "test_prepare_dataset", "test_download_dataset", "test_training", "-v"], "verification.log")
            atomic_json(verification_path, {"passed": True, "device": "cuda", "sources": verified_sources})
        if not (ROOT / "prepared/audit.json").exists():
            run(["prepare_dataset.py", "--raw-root", str(ROOT / "data/final/images"),
                 "--mapping", "object_mapping.csv", "--output", "prepared", "--seed", "42"], "preprocessing.log")
        audit = json.loads((ROOT / "prepared/audit.json").read_text())
        if audit["status"] != "success":
            raise RuntimeError("Preprocessing audit failed; review the saved audit before training")
        args = ["train.py", "--raw-root", str(ROOT / "data/final/images"), "--prepared-dir", str(ROOT / "prepared"),
                "--output", str(ROOT / "runs/convnext_tiny_seed42"), "--seed", "42", "--device", "cuda",
                "--batch-size", "32", "--workers", "2", "--frozen-epochs", "5", "--finetune-epochs", "15"]
        if (ROOT / "runs/convnext_tiny_seed42/config.json").exists():
            args.append("--resume")
        run(args, "training.log")
    finally:
        entries = [(path, path.name) for path in sorted(ROOT.iterdir())
                   if path.name not in {"data", "__pycache__"}]
        entries.extend((path, "acquisition/" + path.name)
                       for name in ("acquisition_manifest.json", "acquisition_inventory.json")
                       if (path := ROOT / "data/final" / name).exists())
        artifacts = ROOT.parent / "milestone1_artifacts.tar.gz"
        package_archive(artifacts, entries)
        summary = experiment_summary()
        split_archive(artifacts, "artifacts", summary)
        atomic_json(ROOT.parent / "milestone1_status.json", summary)


if __name__ == "__main__":
    main()
