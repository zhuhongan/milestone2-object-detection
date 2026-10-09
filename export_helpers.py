"""Run the Colab experiment and atomically package byte-verified export parts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path("/content/milestone2")
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

