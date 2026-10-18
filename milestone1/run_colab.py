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
    archive = f"milestone1_{kind}.tar.gz"
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
    download(f"/content/milestone1_{kind}.tar.gz.manifest.json", manifest_file)
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


def prepare_snapshot_upload(snapshot, destination):
    """Create reusable 32 MiB parts for a local tar containing final/... ."""
    snapshot = Path(snapshot).expanduser().resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    parts_dir = destination / "parts"
    parts_dir.mkdir(exist_ok=True)
    parts, whole, total = [], hashlib.sha256(), 0
    with snapshot.open("rb") as source:
        index = 0
        while block := source.read(PART_BYTES):
            name = f"part-{index:05d}"
            path = parts_dir / name
            digest = hashlib.sha256(block).hexdigest()
            if not valid_file(path, len(block), digest):
                temporary = path.with_name(name + ".partial")
                temporary.write_bytes(block)
                temporary.replace(path)
            parts.append({"name": name, "size_bytes": len(block), "sha256": digest})
            whole.update(block)
            total += len(block)
            index += 1
    if not parts:
        raise RuntimeError("The supplied raw snapshot is empty")
    manifest = {"schema_version": 1, "kind": "raw_upload", "archive_name": snapshot.name,
                "size_bytes": total, "sha256": whole.hexdigest(), "parts": parts}
    manifest_path = destination / "manifest.json"
    temporary = manifest_path.with_name("manifest.json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest_path, manifest


def upload_snapshot(cli, session, snapshot, destination):
    manifest_path, manifest = prepare_snapshot_upload(snapshot, destination)
    ensure_remote_directories(cli, session, destination, ["/content/milestone1_raw_upload.parts"])
    for index, part in enumerate(manifest["parts"]):
        cli("upload", "-s", session, str(manifest_path.parent / "parts" / part["name"]),
            "/content/milestone1_raw_upload.parts/" + part["name"])
        print(f"Uploaded raw snapshot part {index + 1}/{len(manifest['parts'])}", flush=True)
    cli("upload", "-s", session, str(manifest_path), "/content/milestone1_raw_upload.manifest.json")
    cli("exec", "-s", session, "-f", str(PROJECT / "colab_receive_raw.py"), "--timeout", "600")
    # Do not trust exec exit status: require a receiver completion receipt.
    receipt_path = manifest_path.parent / "import_receipt.json"
    receipt_path.unlink(missing_ok=True)
    cli("download", "-s", session, "/content/milestone1_raw_import_complete.json", str(receipt_path))
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("complete") is not True or receipt.get("archive_sha256") != manifest["sha256"]:
        raise RuntimeError("Colab did not confirm a verified raw snapshot import")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=["T4", "L4", "A100", "H100"], default="L4")
    parser.add_argument("--session", help="Unique session name; existing names are rejected for new runs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-snapshot", type=Path, help="Upload a local verified raw tar containing final/... instead of downloading from Drive on Colab")
    parser.add_argument("--recover", action="store_true", help="Retrieve an existing run and stop it after verified recovery")
    args = parser.parse_args()
    if args.recover and (not args.session or not args.output):
        parser.error("--recover requires both --session and the original --output directory")
    if args.recover and args.raw_snapshot:
        parser.error("--raw-snapshot cannot be combined with --recover")
    if args.raw_snapshot and not args.raw_snapshot.expanduser().is_file():
        parser.error("--raw-snapshot must name an existing local .tar.gz snapshot")
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    session = args.session or "milestone1-convnext-" + identifier
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session):
        parser.error("Session names must use 1-100 letters, digits, underscores or hyphens")
    output = (args.output or PROJECT / "artifacts" / identifier).expanduser().resolve()
    config = PROJECT / ".colab-sessions.json"
    base = ["colab", "--config", str(config)]
    mapping = PROJECT / "metadata/object_mapping.csv"
    provenance = mapping.with_suffix(".provenance.json")
    if not args.recover:
        if not mapping.is_file() or not provenance.is_file():
            parser.error("Build metadata/object_mapping.csv from the final Sheet before allocating a GPU.")
        proof = json.loads(provenance.read_text())
        if not proof.get("verified_against_sheet") or proof.get("mapping_sha256") != sha256(mapping):
            parser.error("The verified mapping snapshot changed. Rebuild from the official Sheet.")
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        parser.error("Recovery output directory does not exist")
    started = args.recover
    protect_results = args.recover
    exports_verified = False

    def cli(*arguments):
        run_cli(base, *arguments, cwd=PROJECT)

    def download(remote, destination, size=None, digest=None):
        temporary = destination.with_name(destination.name + ".partial")
        for attempt in range(3):
            try:
                # CLI downloads are base64 JSON in memory. Each part is capped at 32 MiB.
                result = subprocess.run([*base, "download", "-s", session, remote, str(temporary)],
                                        cwd=PROJECT, capture_output=True, text=True, timeout=300)
                if result.returncode:
                    # CLI HTTP errors can contain proxy tokens; do not replay raw errors.
                    raise RuntimeError(f"Colab download returned exit {result.returncode}")
                if size is not None and not valid_file(temporary, size, digest):
                    raise RuntimeError("Downloaded part checksum or size mismatch")
                temporary.replace(destination)
                return
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                if attempt == 2:
                    raise RuntimeError(f"Could not retrieve {PurePosixPath(remote).name} after 3 attempts: {error}") from None
                print(f"Retrying {PurePosixPath(remote).name} ({attempt + 2}/3)...", flush=True)
                time.sleep(2 ** attempt)

    def recover_exports():
        recovered, failures = {}, []
        for kind, local in (("artifacts", "experiment.tar.gz"), ("raw", "raw_snapshot.tar.gz")):
            try:
                recovered[kind] = retrieve_archive(download, output, kind, local)[0]
            except Exception as error:
                failures.append(f"{kind}: {error}")
        if failures:
            raise RuntimeError("Export retrieval incomplete: " + "; ".join(failures))
        return recovered

    try:
        if not args.recover:
            # The CLI silently overwrites duplicate names. Serialize local launches
            # and reject name reuse before allocating anything.
            with (PROJECT / ".colab-launch.lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                sessions = json.loads(config.read_text()) if config.exists() else {}
                if session in sessions:
                    raise RuntimeError(f"Session {session!r} already exists; choose a new name or --recover")
                cli("new", "-s", session, "--gpu", args.gpu)
                started = True
            (output / "session_receipt.json").write_text(json.dumps({"session": session, "gpu": args.gpu,
                "output": str(output), "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
            ensure_remote_directories(cli, session, output / "setup", ["/content/milestone1"])
            files = ["train.py", "prepare_dataset.py", "download_dataset.py", "test_training.py",
                     "test_prepare_dataset.py", "test_download_dataset.py", "colab_prepare_train.py",
                     "colab_download.py", "colab_probe.py", "colab_receive_raw.py", "run_colab.py"]
            for name in files:
                cli("upload", "-s", session, str(PROJECT / name), "/content/milestone1/" + name)
            cli("upload", "-s", session, str(PROJECT / "download_dataset.py"), "/content/download_dataset.py")
            cli("upload", "-s", session, str(mapping), "/content/milestone1/object_mapping.csv")
            cli("upload", "-s", session, str(provenance), "/content/milestone1/object_mapping.provenance.json")
            cli("exec", "-s", session, "-f", str(PROJECT / "colab_probe.py"), "--timeout", "120")
            if args.raw_snapshot:
                upload_snapshot(cli, session, args.raw_snapshot, output / "raw_upload")
            else:
                cli("exec", "-s", session, "-f", str(PROJECT / "colab_download.py"), "--timeout", "3600")
            protect_results = True
            try:
                cli("exec", "-s", session, "-f", str(PROJECT / "colab_prepare_train.py"), "--timeout", "14400")
            finally:
                recovered = recover_exports()
                exports_verified = True
        else:
            recovered = recover_exports()
            exports_verified = True
        unpack(recovered["artifacts"], output / "experiment")
        # Colab exec can return zero for remote exceptions. Verify our own marker.
        completed = output / "experiment/runs/convnext_tiny_seed42/completed.json"
        if not completed.is_file() or not json.loads(completed.read_text()).get("pretrained"):
            raise RuntimeError(f"Experiment did not finish; verified audits/checkpoints were saved to {output}")
        print(f"Saved verified raw snapshot, checkpoints, plots, metrics, logs, and sources to {output}")
    finally:
        if started:
            if protect_results and not exports_verified:
                recovery = shlex.join([sys.executable, str(PROJECT / "run_colab.py"), "--recover",
                                       "--session", session, "--output", str(output)])
                stop = shlex.join([*base, "stop", "-s", session])
                print(f"\nGPU session {session!r} is retained because its exports are not yet verified locally. "
                      "The runtime may continue consuming Colab resources. Recover promptly; recovery stops it after verification.\n"
                      f"Recovery command:\n{recovery}\n"
                      f"To deliberately discard remaining remote data and stop instead:\n{stop}", file=sys.stderr)
            else:
                cli("stop", "-s", session)


if __name__ == "__main__":
    main()
