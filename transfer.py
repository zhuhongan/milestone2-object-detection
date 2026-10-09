#!/usr/bin/env python3
"""Run Milestone 1 with Colab CLI, verify exports, then release the owned GPU.

Use --recover --session NAME --output DIR to retry artifact retrieval from a
retained runtime without allocating another GPU or running training again.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

PROJECT = Path(__file__).resolve().parent
PART_BYTES = 32 * 1024 * 1024


def sanitize_cli_output(value):
    """Keep useful CLI errors/progress without replaying proxy credentials."""
    value = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)
    value = re.sub(r"https?://[^\s<>\"']+", "[URL omitted]", value)
    value = re.sub(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;\"']+",
                   r"\1[REDACTED]", value)
    value = re.sub(r"(?i)((?:colab-runtime-proxy-token|access_token|refresh_token|id_token|token|api_key|key)"
                   r"[\"']?\s*[:=]\s*[\"']?)[^\s&\"',<>}]+", r"\1[REDACTED]", value)
    return value


def run_cli(base, *arguments, cwd=PROJECT):
    """Stream captured and sanitized output, including errors printed to stdout."""
    process = subprocess.Popen([*base, *arguments], cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(sanitize_cli_output(line), end="", flush=True)
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Colab {arguments[0]} failed with exit code {returncode}")


def ensure_remote_directories(cli, session, destination, directories):
    """The Contents upload API requires parent directories to exist first."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    script = destination / "create_remote_directories.py"
    script.write_text("from pathlib import Path\n" +
                      "\n".join(f"Path({path!r}).mkdir(parents=True, exist_ok=True)" for path in directories) +
                      "\nprint('Remote upload directories ready', flush=True)\n")
    # This Colab CLI supports --file (and stdin), but has no -c/--code option.
    cli("exec", "-s", session, "-f", str(script), "--timeout", "60")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_file(path, size, digest):
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def validate_manifest(manifest, kind):
    archive = f"milestone2_{kind}.tar.gz"
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("kind") != kind or manifest.get("archive_name") != archive):
        raise RuntimeError(f"Invalid {kind} export manifest")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise RuntimeError("Export manifest has no parts")
    expected_prefix = f"/content/{archive}.parts"
    total = 0
    for index, part in enumerate(parts):
        name = f"part-{index:05d}"
        if (part.get("name") != name or part.get("remote_path") != f"{expected_prefix}/{name}"
                or not isinstance(part.get("size_bytes"), int)
                or not 0 < part["size_bytes"] <= PART_BYTES
                or not re.fullmatch(r"[a-f0-9]{64}", part.get("sha256", ""))):
            raise RuntimeError(f"Invalid export part {index}")
        total += part["size_bytes"]
    if total != manifest.get("size_bytes") or not re.fullmatch(r"[a-f0-9]{64}", manifest.get("sha256", "")):
        raise RuntimeError("Export manifest total or checksum is invalid")
    return manifest


def unpack(source, destination):
    destination = destination.resolve()
    destination.mkdir(exist_ok=True)
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        for item in members:
            target = (destination / item.name).resolve()
            if destination not in target.parents or not (item.isfile() or item.isdir()):
                raise RuntimeError(f"Unsafe archive member: {item.name}")
        # Copy only the verified ordinary files/directories. This is portable
        # to Python versions predating tarfile's data extraction filter and
        # never restores links, device nodes, ownership, or executable modes.
        for item in members:
            target = destination / item.name
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)


def retrieve_archive(download, output, kind, local_name):
    manifest_file = output / f"{kind}.manifest.json"
    download(f"/content/milestone2_{kind}.tar.gz.manifest.json", manifest_file)
    manifest = validate_manifest(json.loads(manifest_file.read_text()), kind)
    archive_path = output / local_name
    if valid_file(archive_path, manifest["size_bytes"], manifest["sha256"]):
        return archive_path, manifest
    parts_dir = output / (local_name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    pending = []
    for index, part in enumerate(manifest["parts"]):
        local = parts_dir / part["name"]
        if not valid_file(local, part["size_bytes"], part["sha256"]):
            pending.append((index, part, local))
    # The CLI materializes base64 JSON in memory. Three 32 MiB transfers keep
    # concurrency bounded while overlapping independent network requests.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download, part["remote_path"], local, part["size_bytes"], part["sha256"]): index
                   for index, part, local in pending}
        try:
            for future in as_completed(futures):
                future.result()
                print(f"Verified {kind} export part {futures[future] + 1}/{len(manifest['parts'])}", flush=True)
        except BaseException:
            # Preserve validated parts, cancel work that has not started, and
            # let only the at-most-three bounded in-flight callbacks finish.
            for future in futures:
                future.cancel()
            raise
    temporary = archive_path.with_name(archive_path.name + ".partial")
    with temporary.open("wb") as destination:
        for part in manifest["parts"]:
            with (parts_dir / part["name"]).open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    if not valid_file(temporary, manifest["size_bytes"], manifest["sha256"]):
        raise RuntimeError(f"Reassembled {kind} archive checksum mismatch")
    temporary.replace(archive_path)
    return archive_path, manifest

