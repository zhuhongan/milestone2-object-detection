#!/usr/bin/env python3
"""Validate immutable class folders and make reproducible, leakage-free manifests.

Mapping CSV columns: folder,object_id,object_name. Object IDs remain strings;
class_index is a separate contiguous model target. No source file is changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import struct
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, __version__ as pillow_version


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".mpo", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = [
    "path", "folder", "object_id", "object_name", "class_index", "sha256",
    "pixel_sha256", "width", "height", "format", "source_frames", "selected_frame",
]
EXCLUSION_FIELDS = ["path", "reason", "details", "duplicate_of"]


class DatasetPreparationError(ValueError):
    """The input cannot produce trustworthy manifests; consult audit.json."""


class UnsupportedMultiframeImage(ValueError):
    """Only JPEG/MPO has an explicit primary-frame classification policy."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mapping(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = ("folder", "object_id", "object_name")
        if not set(required).issubset(reader.fieldnames or []):
            raise DatasetPreparationError("Mapping CSV must contain folder,object_id,object_name columns.")
        rows = []
        for line, original in enumerate(reader, start=2):
            if not any(value and str(value).strip() for value in original.values()):
                continue
            row = {key: (original.get(key) or "").strip() for key in required}
            if not all(row.values()):
                raise DatasetPreparationError(f"Mapping line {line} has an empty required value.")
            folder = row["folder"]
            if folder in {".", ".."} or "/" in folder or "\\" in folder:
                raise DatasetPreparationError(f"Mapping folder must name one direct child folder: {folder!r}.")
            rows.append(row)
    if not rows:
        raise DatasetPreparationError("Mapping CSV has no classes.")
    for key in ("folder", "object_id"):
        repeated = sorted(value for value, count in Counter(row[key] for row in rows).items() if count > 1)
        if repeated:
            raise DatasetPreparationError(f"Mapping {key} values must be unique; repeated: {repeated}.")
    rows.sort(key=lambda row: (row["object_id"], row["folder"]))
    return [dict(row, class_index=index) for index, row in enumerate(rows)]


def inspect_image(path: Path) -> dict[str, Any]:
    """Fully decode images and hash EXIF-oriented RGB pixels including dimensions.

    Pillow's size threshold is a heuristic, not a corruption test. Disable it for
    this known dataset, then fully verify/decode. Process RGB hash data in strips
    so hashing does not allocate another full-image bytes buffer.
    """
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image_format = image.format
            frames = getattr(image, "n_frames", 1)
            if frames != 1 and image_format not in {"JPEG", "MPO"}:
                raise UnsupportedMultiframeImage(f"{image_format} contains {frames} frames; primary-frame policy is defined only for JPEG/MPO.")
            # Phone JPEG files can be MPO containers with a primary photo and
            # auxiliary frames such as an HDR gain map. Classify the full primary
            # photo exactly as Image.open(...), EXIF transpose, RGB in training.
            image.seek(0)
            image.load()
            oriented = ImageOps.exif_transpose(image)
            rgb = oriented.convert("RGB")
            try:
                width, height = rgb.size
                digest = hashlib.sha256(b"EXIF-oriented-RGB-v1\0" + struct.pack(">QQ", width, height))
                strip_height = max(1, (1024 * 1024) // max(1, width * 3))
                for top in range(0, height, strip_height):
                    strip = rgb.crop((0, top, width, min(height, top + strip_height)))
                    try:
                        digest.update(strip.tobytes())
                    finally:
                        strip.close()
                return {
                    "width": width, "height": height, "format": image_format,
                    "pixel_sha256": digest.hexdigest(), "source_frames": frames,
                    "selected_frame": 0,
                }
            finally:
                rgb.close()
                oriented.close()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def split_counts(count: int) -> dict[str, int]:
    """Hamilton allocation using integer arithmetic; ties use train, val, test."""
    weights = (8, 1, 1)
    counts = [count * weight // 10 for weight in weights]
    remainder_order = sorted(range(3), key=lambda index: (-(count * weights[index] % 10), index))
    for index in remainder_order[: count - sum(counts)]:
        counts[index] += 1
    return dict(zip(SPLITS, counts))


def _atomic_text(path: Path, value: str) -> None:
    # Exclusive temporary creation avoids following preexisting file symlinks.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=".prepare-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(value)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def prepare_dataset(raw_root: Path | str, mapping: Path | str, output: Path | str, seed: int = 42) -> dict[str, Any]:
    raw_root, mapping, output = Path(raw_root).resolve(), Path(mapping).resolve(), Path(output).resolve()
    if not raw_root.is_dir():
        raise DatasetPreparationError(f"Raw dataset directory does not exist: {raw_root}")
    if output == raw_root or raw_root in output.parents:
        raise DatasetPreparationError("Output must be outside the locked raw dataset. No raw files may be changed.")
    if output == mapping or output in mapping.parents:
        # Reading a mapping from an existing output directory is harmless, but its
        # name must not collide with files owned by this preparation command.
        if mapping.name in {"audit.json", "exclusions.csv", "class_mapping.json", "conflicts.json", "train.csv", "val.csv", "test.csv"}:
            raise DatasetPreparationError("Mapping path collides with a generated artifact.")
    output.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "raw_root": str(raw_root),
        "mapping_path": str(mapping),
        "seed": seed,
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "split_method": "Per-class Python random.Random(seed).shuffle; Hamilton largest remainder with train/val/test tie order; require nonempty splits per class.",
        "ordering": "Classes sorted lexicographically by string object_id; files sorted by relative POSIX path.",
        "pillow_version": pillow_version,
        "pixel_hash_policy": "SHA256(EXIF-oriented-RGB-v1 NUL + big-endian uint64 width,height + row-major RGB bytes)",
        "source_policy": "Read-only raw files; all exclusions are manifest exclusions, never file deletions.",
        "dimensions_policy": "Retain all fully decodable primary-image dimensions; training performs online architecture-specific resizing.",
        "frame_policy": "JPEG/MPO uses primary frame 0, including phone JPEGs with auxiliary HDR/gain-map frames. Validate/decode, EXIF-orient, and RGB-hash frame 0 only, matching training. Other multiframe formats require an explicit separate policy and are excluded with a reason.",
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }
    exclusions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    _write_json(output / "audit.json", audit)

    def exclude(path: str, reason: str, details: str = "", duplicate_of: str = "") -> None:
        exclusions.append({"path": path, "reason": reason, "details": details, "duplicate_of": duplicate_of})

    try:
        classes = read_mapping(mapping)
        audit["mapping_sha256"] = sha256_file(mapping)
        by_folder = {row["folder"]: row for row in classes}
        for folder in by_folder:
            directory = raw_root / folder
            if not directory.is_dir() or directory.is_symlink():
                raise DatasetPreparationError(f"Mapped class directory is missing or is a symlink: {folder}")

        paths: list[Path] = []
        for current, dirs, files in os.walk(raw_root, followlinks=False):
            directory = Path(current)
            for name in list(dirs):
                child = directory / name
                if child.is_symlink():
                    reason = "symlink_outside_raw_root" if not child.resolve().is_relative_to(raw_root) else "symlink_not_followed"
                    exclude(child.relative_to(raw_root).as_posix(), reason, "Directory symlinks are not traversed.")
                    dirs.remove(name)
            paths.extend(directory / name for name in files)
        paths.sort(key=lambda path: path.relative_to(raw_root).as_posix())

        valid: list[dict[str, Any]] = []
        unmapped_images: list[str] = []
        for path in paths:
            relative = path.relative_to(raw_root).as_posix()
            if path.is_symlink():
                reason = "symlink_outside_raw_root" if not path.resolve().is_relative_to(raw_root) else "symlink_not_followed"
                exclude(relative, reason, "File symlinks are not followed.")
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                exclude(relative, "unsupported_nonimage_file", "Extension is outside the supported image formats; inspect separately if this is an image.")
                continue
            parts = path.relative_to(raw_root).parts
            if len(parts) < 2 or parts[0] not in by_folder:
                unmapped_images.append(relative)
                continue
            try:
                metadata = inspect_image(path)
                raw_hash = sha256_file(path)
            except UnsupportedMultiframeImage as error:
                exclude(relative, "unsupported_multiframe_format", str(error))
                continue
            except (OSError, ValueError, SyntaxError) as error:
                exclude(relative, "invalid_or_unsupported_image", f"{type(error).__name__}: {error}")
                continue
            valid.append(dict(by_folder[parts[0]], path=relative, sha256=raw_hash, **metadata))

        audit["scanned_files"] = len(paths)
        audit["valid_images_before_deduplication"] = len(valid)
        audit["valid_multiframe_jpeg_mpo_primary_images"] = sum(row["source_frames"] > 1 for row in valid)
        audit["valid_format_counts"] = dict(sorted(Counter(row["format"] for row in valid).items()))
        audit["unmapped_image_paths"] = unmapped_images
        if unmapped_images:
            raise DatasetPreparationError("Image files have no official folder/Object ID mapping; see audit.json unmapped_image_paths.")

        # Union both exact byte identities and exact decoded pixel identities.
        # Selecting the lexicographically first path is deterministic and retains
        # originals without reencoding. Cross-label unions are fatal.
        parents = list(range(len(valid)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first, second = find(first), find(second)
            parents[max(first, second)] = min(first, second)

        for field in ("sha256", "pixel_sha256"):
            seen: dict[str, int] = {}
            for index, row in enumerate(valid):
                if row[field] in seen:
                    union(index, seen[row[field]])
                else:
                    seen[row[field]] = index
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, row in enumerate(valid):
            groups[find(index)].append(row)
        retained = []
        for group in groups.values():
            if len({row["object_id"] for row in group}) > 1:
                conflicts.append({"reason": "identical_image_has_multiple_object_ids", "images": group})
                continue
            canonical = group[0]
            retained.append(canonical)
            for row in group[1:]:
                reason = "exact_file_duplicate" if row["sha256"] == canonical["sha256"] else "exact_rgb_pixel_duplicate"
                exclude(row["path"], reason, "Removed from all split manifests before splitting.", canonical["path"])
        if conflicts:
            raise DatasetPreparationError("Identical images have conflicting Object IDs. Resolve label policy without modifying the locked raw dataset; see conflicts.json.")

        result: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
        per_class = []
        rng = random.Random(seed)
        for class_row in classes:
            rows = [row for row in retained if row["class_index"] == class_row["class_index"]]
            counts = split_counts(len(rows))
            if not all(counts.values()):
                raise DatasetPreparationError(f"Object ID {class_row['object_id']!r} has {len(rows)} retained images, producing {counts}. Every class must have a nonempty train/val/test allocation.")
            rng.shuffle(rows)
            cursor = 0
            for name in SPLITS:
                result[name].extend(rows[cursor:cursor + counts[name]])
                cursor += counts[name]
            per_class.append(dict(class_row, retained=len(rows), **counts))

        for field in ("path", "sha256", "pixel_sha256"):
            observed: set[str] = set()
            for name in SPLITS:
                hashes = {row[field] for row in result[name]}
                if observed.intersection(hashes):
                    raise DatasetPreparationError(f"Internal leakage check failed for {field}.")
                observed.update(hashes)
        for name in SPLITS:
            result[name].sort(key=lambda row: (row["class_index"], row["path"]))
            _write_csv(output / f"{name}.csv", result[name], MANIFEST_FIELDS)
        _write_json(output / "class_mapping.json", classes)
        audit.update({
            "status": "success",
            "retained_images": len(retained),
            "split_counts": {name: len(result[name]) for name in SPLITS},
            "per_class": per_class,
            "leakage_checks": {"paths_disjoint": True, "raw_sha256_disjoint": True, "rgb_pixel_sha256_disjoint": True},
            "manifest_sha256": {f"{name}.csv": sha256_file(output / f"{name}.csv") for name in SPLITS},
            "class_mapping_sha256": sha256_file(output / "class_mapping.json"),
        })
        return audit
    except Exception as error:
        audit.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        exclusions.sort(key=lambda row: row["path"])
        audit["exclusion_counts"] = dict(sorted(Counter(row["reason"] for row in exclusions).items()))
        audit["exclusion_total"] = len(exclusions)
        audit["cross_label_duplicate_groups"] = len(conflicts)
        _write_csv(output / "exclusions.csv", exclusions, EXCLUSION_FIELDS)
        _write_json(output / "conflicts.json", conflicts)
        _write_json(output / "audit.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        audit = prepare_dataset(args.raw_root, args.mapping, args.output, args.seed)
    except (DatasetPreparationError, OSError) as error:
        parser.exit(2, f"Preparation stopped: {error}\n")
    print(json.dumps({key: audit[key] for key in ("status", "retained_images", "split_counts", "exclusion_counts")}, indent=2))


if __name__ == "__main__":
    main()
