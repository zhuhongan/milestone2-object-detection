"""Verify a chunk-uploaded immutable raw snapshot and safely install it on Colab.

Run through `colab exec -f colab_receive_raw.py` after uploading the manifest to
/content/milestone1_raw_upload.manifest.json and its listed parts. This script
does not access Drive, overwrite an existing final snapshot, or execute dataset
contents. Failed imports remain in a separate staging directory for diagnosis.
"""
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tempfile

BASE = Path("/content")
PART_BYTES = 32 * 1024 * 1024
FOLDER_ID = "1GKLyP17tm5JC0XENv25GeoGgYynm97LS"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(snapshot):
    manifest = json.loads((snapshot / "acquisition_manifest.json").read_text())
    if manifest.get("complete") is not True or manifest.get("source_folder_id") != FOLDER_ID:
        raise RuntimeError("Raw snapshot is incomplete or has the wrong dataset source")
    files = manifest["files"]
    if len(files) != manifest["file_count"] or not files:
        raise RuntimeError("Acquisition manifest file count is invalid")
    paths = set()
    image_root = (snapshot / "images").resolve()
    for item in files:
        relative = Path(item["relative_path"])
        path = (image_root / relative).resolve()
        if relative.is_absolute() or image_root not in path.parents or str(relative) in paths:
            raise RuntimeError("Unsafe or duplicate acquisition path")
        paths.add(str(relative))
        if path.stat().st_size != int(item["size"]) or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Snapshot byte verification failed: {relative}")
    actual = {str(path.relative_to(image_root)) for path in image_root.rglob("*") if path.is_file()}
    if actual != paths:
        raise RuntimeError("Snapshot image tree does not match its complete acquisition manifest")
    return len(files)


def main(base=BASE):
    base = Path(base).resolve()
    final = base / "milestone1/data/final"
    if final.exists():
        raise RuntimeError("Final raw snapshot already exists; refusing to overwrite it")
    manifest = json.loads((base / "milestone1_raw_upload.manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "raw_upload":
        raise RuntimeError("Invalid upload manifest")
    parts = manifest.get("parts", [])
    if not parts or not re.fullmatch(r"[a-f0-9]{64}", manifest.get("sha256", "")):
        raise RuntimeError("Missing upload parts or archive SHA-256")
    whole, total = hashlib.sha256(), 0
    archive = base / "milestone1_raw_upload.tar.gz"
    temporary = archive.with_name(archive.name + ".partial")
    with temporary.open("wb") as output:
        for index, item in enumerate(parts):
            name = f"part-{index:05d}"
            if (item.get("name") != name or not isinstance(item.get("size_bytes"), int)
                    or not 0 < item["size_bytes"] <= PART_BYTES
                    or not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", ""))):
                raise RuntimeError(f"Invalid upload part {index}")
            path = base / "milestone1_raw_upload.parts" / name
            if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
                raise RuntimeError(f"Uploaded part checksum mismatch: {name}")
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    whole.update(block)
                    total += len(block)
    if total != manifest.get("size_bytes") or whole.hexdigest() != manifest["sha256"]:
        raise RuntimeError("Reassembled raw archive checksum mismatch")
    temporary.replace(archive)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="local-import-", dir=final.parent)).resolve()
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            target = (staging / member.name).resolve()
            if (staging not in target.parents or not member.name.startswith("final/") and member.name != "final"
                    or not (member.isfile() or member.isdir())):
                raise RuntimeError(f"Unsafe raw archive member: {member.name}")
        source.extractall(staging, filter="data")
    count = verify_snapshot(staging / "final")
    if final.exists():
        raise RuntimeError("Final raw snapshot appeared during verification; refusing overwrite")
    (staging / "final").rename(final)
    receipt = {"complete": True, "file_count": count, "archive_sha256": manifest["sha256"],
               "raw_root": str(final / "images"), "source_folder_id": FOLDER_ID}
    (base / "milestone1_raw_import_complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
