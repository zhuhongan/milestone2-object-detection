#!/usr/bin/env python3
"""Acquire a complete, byte-verified, local snapshot without modifying Google Drive.

Only Drive v3 GET requests are made. Lists are paginated, including each child
folder; an incomplete listing or failed checksum aborts the snapshot. The public
mode uses the same public API configuration exposed by Google's Drive web page.
It does not read browser cookies, Colab credentials, or account tokens. Set
DRIVE_API_KEY to supply your own public-data API key instead. For private data,
--auth adc uses existing Application Default Credentials with a read-only scope.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import uuid

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_FOLDER = "1GKLyP17tm5JC0XENv25GeoGgYynm97LS"
API = "https://www.googleapis.com/drive/v3"
FOLDER_MIME = "application/vnd.google-apps.folder"
FIELDS = "id,name,mimeType,size,md5Checksum,modifiedTime,version,shortcutDetails"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_name(name):
    # Keep the official Object ID folders intact, and reject path traversal.
    if name in {"", ".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise RuntimeError(f"Unsafe Drive filename: {name!r}")
    return name


def new_session():
    session = requests.Session()
    # A 403 is an access/rate denial, never an automatic retry condition.
    retry = Retry(total=6, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["Referer"] = "https://drive.google.com/"
    return session


class DriveReader:
    def __init__(self, folder_id, auth="public"):
        self.folder_id = folder_id
        self.auth = auth
        self.api_key = None
        self.credentials = None
        self.cancelled = threading.Event()
        self.local = threading.local()
        if auth == "adc":
            import google.auth
            self.credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/drive.readonly"])
        else:
            supplied_key = os.environ.get("DRIVE_API_KEY")
            if supplied_key:
                keys = [supplied_key]
            else:
                response = new_session().get(
                    f"https://drive.google.com/drive/folders/{folder_id}", timeout=60)
                if response.status_code != 200:
                    raise RuntimeError(f"Public Drive page returned HTTP {response.status_code}")
                keys = list(dict.fromkeys(re.findall(r"AIza[0-9A-Za-z_-]{35}", response.text)))
            for key in keys:
                try:
                    response = new_session().get(f"{API}/files", params={
                        "key": key, "q": f"'{folder_id}' in parents and trashed = false",
                        "fields": "files(id)", "pageSize": 1,
                    }, timeout=60)
                except requests.RequestException as exc:
                    raise RuntimeError(f"Public API availability check failed "
                                       f"({type(exc).__name__})") from None
                if response.status_code == 200:
                    self.api_key = key
                    break
            if not self.api_key:
                raise RuntimeError("Cannot read complete public Drive listing. Supply DRIVE_API_KEY "
                                   "or use --auth adc with read-only Drive access. "
                                   "Refusing an incomplete HTML/gdown folder listing.")

    def session(self):
        if not hasattr(self.local, "session"):
            if self.auth == "adc":
                from google.auth.transport.requests import AuthorizedSession
                self.local.session = AuthorizedSession(self.credentials)
            else:
                self.local.session = new_session()
        return self.local.session

    def get(self, route, params=None, stream=False):
        if self.cancelled.is_set():
            raise RuntimeError("Acquisition cancelled after another download failed")
        params = dict(params or {})
        if self.api_key:
            params["key"] = self.api_key
        try:
            response = self.session().get(f"{API}/{route}", params=params,
                                          stream=stream, timeout=(30, 180))
        except requests.RequestException as exc:
            # requests exception URLs may include API keys; do not print them.
            raise RuntimeError(f"Drive GET {route} failed ({type(exc).__name__})") from None
        if not response.ok:
            status = response.status_code
            denial = ""
            if status == 403:
                body = response.text.lower()
                if "automated queries" in body:
                    denial = "; Google reports automated-query rate protection"
                else:
                    denial = "; request denied (no automatic retry)"
            response.close()
            raise RuntimeError(f"Drive GET {route} returned HTTP {status}{denial}")
        return response

    def children(self, folder_id):
        files = []
        token = None
        seen_tokens = set()
        while True:
            params = {"q": f"'{folder_id}' in parents and trashed = false",
                      "fields": f"nextPageToken,incompleteSearch,files({FIELDS})",
                      "pageSize": 1000, "supportsAllDrives": "true",
                      "includeItemsFromAllDrives": "true"}
            if token:
                params["pageToken"] = token
            data = self.get("files", params).json()
            if data.get("incompleteSearch"):
                raise RuntimeError(f"Drive returned incompleteSearch for folder {folder_id}")
            files.extend(data.get("files", []))
            token = data.get("nextPageToken")
            if not token:
                break
            if token in seen_tokens:
                raise RuntimeError("Drive pagination repeated a token")
            seen_tokens.add(token)
        ids = [file["id"] for file in files]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"Drive listing repeated file IDs in {folder_id}")
        return sorted(files, key=lambda file: (file["name"], file["id"]))

    def inventory(self, workers=8):
        root = self.get(f"files/{self.folder_id}", {"fields": FIELDS,
                                                   "supportsAllDrives": "true"}).json()
        if root["mimeType"] != FOLDER_MIME:
            raise RuntimeError("Dataset source is not a Drive folder")
        pending = [(self.folder_id, "")]
        all_files, folders, excluded = [], [], []
        visited = set()
        while pending:
            batch, pending = pending, []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self.children, fid): (fid, parent) for fid, parent in batch}
                for future in as_completed(futures):
                    fid, parent = futures[future]
                    if fid in visited:
                        raise RuntimeError("Repeated/cyclic Drive folder in inventory")
                    visited.add(fid)
                    children = future.result()
                    names = set()
                    for child in children:
                        name = safe_name(child["name"])
                        # Drive allows same-name files. Preserve each without overwriting.
                        if name.casefold() in names:
                            suffix = Path(name).suffix
                            name = f"{Path(name).stem}__drive_{child['id']}{suffix}"
                        if name.casefold() in names:
                            raise RuntimeError("Unresolvable same-name Drive files")
                        names.add(name.casefold())
                        path = str(Path(parent) / name)
                        item = {**child, "relative_path": path, "parent_id": fid}
                        mime = child["mimeType"]
                        if mime == FOLDER_MIME:
                            folders.append(item)
                            pending.append((child["id"], path))
                        elif mime == "application/vnd.google-apps.shortcut":
                            raise RuntimeError(f"Drive shortcut needs explicit resolution: {path}")
                        elif mime.startswith("application/vnd.google-apps."):
                            excluded.append({**item, "reason": "Native Google document; not an image byte file"})
                        else:
                            if "size" not in child or "md5Checksum" not in child:
                                raise RuntimeError(f"Missing size or MD5 metadata for {path}")
                            all_files.append(item)
        all_files.sort(key=lambda item: item["relative_path"])
        folders.sort(key=lambda item: item["relative_path"])
        return {"schema_version": 1, "source_folder_id": self.folder_id,
                "source_url": f"https://drive.google.com/drive/folders/{self.folder_id}",
                "source_name": root["name"], "inventory_created_utc": utc_now(),
                "listing_method": "Google Drive v3 files.list with full pagination",
                "remote_operations": "GET only", "folders": folders, "files": all_files,
                "excluded_native_documents": excluded, "file_count": len(all_files),
                "total_bytes": sum(int(item["size"]) for item in all_files)}


def hashes(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            sha.update(chunk)
            md5.update(chunk)
    return size, md5.hexdigest(), sha.hexdigest()


def download_one(reader, staging, item):
    path = staging / "images" / item["relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".download")
        # Recover only incomplete temporary data owned by this acquisition.
        with reader.get(f"files/{item['id']}", {"alt": "media", "supportsAllDrives": "true"},
                        stream=True) as response:
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    cancelled = getattr(reader, "cancelled", None)
                    if cancelled is not None and cancelled.is_set():
                        raise RuntimeError("Acquisition cancelled; verified cache was preserved")
                    if chunk:
                        output.write(chunk)
        size, md5, sha = hashes(temporary)
        if size != int(item["size"]) or md5 != item["md5Checksum"]:
            raise RuntimeError(f"Downloaded size/MD5 mismatch for {item['relative_path']}")
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
        temporary.rename(path)
    else:
        size, md5, sha = hashes(path)
        if size != int(item["size"]) or md5 != item["md5Checksum"]:
            raise RuntimeError(f"Cached file mismatch for {item['relative_path']}; create a new destination")
    return {**item, "local_size_bytes": size, "sha256": sha, "md5_verified": True}


def write_new_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)
        output.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-id", default=DEFAULT_FOLDER)
    parser.add_argument("--dest", type=Path,
                        default=Path("data/raw") / ("final-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--auth", choices=["public", "adc"], default="public")
    parser.add_argument("--inventory-only", type=Path, help="Write complete inventory JSON, without downloading files")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.folder_id):
        parser.error("Invalid Drive folder ID")
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    if args.dest.exists() and not args.inventory_only:
        parser.error(f"Destination already exists; finalized snapshots are never overwritten: {args.dest}")
    reader = DriveReader(args.folder_id, args.auth)
    print("Enumerating complete remote folder tree with pagination...", flush=True)
    current = reader.inventory(args.workers)
    print(f"Inventory: {len(current['folders'])} folders, {current['file_count']} files, "
          f"{current['total_bytes'] / 1024**3:.2f} GiB", flush=True)
    if not current["files"]:
        raise RuntimeError("Empty dataset; refusing to create snapshot")
    if args.inventory_only:
        write_new_json(args.inventory_only, current)
        print(f"Inventory saved: {args.inventory_only.resolve()}", flush=True)
        return
    staging = args.dest.with_name(args.dest.name + ".partial")
    staging.mkdir(parents=True, exist_ok=True)
    # Unix advisory lock prevents two resumptions writing the same staging tree.
    acquisition_lock = (staging / ".acquisition.lock").open("a")
    try:
        fcntl.flock(acquisition_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another acquisition is already writing this staging directory") from None
    inventory_file = staging / "acquisition_inventory.json"
    if inventory_file.exists():
        inventory = json.loads(inventory_file.read_text())
        # Ignore the acquisition time, but demand that source IDs/bytes match.
        for key in ["source_folder_id", "files", "folders", "excluded_native_documents"]:
            if inventory[key] != current[key]:
                raise RuntimeError("Remote dataset changed since partial snapshot; choose a new --dest")
    else:
        inventory = current
        write_new_json(inventory_file, inventory)
    for folder in inventory["folders"]:
        (staging / "images" / folder["relative_path"]).mkdir(parents=True, exist_ok=True)
    results = []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    failed = False
    try:
        futures = [pool.submit(download_one, reader, staging, item) for item in inventory["files"]]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0 or len(results) == len(futures):
                print(f"Verified {len(results)}/{len(futures)} files", flush=True)
    except BaseException as error:
        failed = True
        reader.cancelled.set()
        print(f"Acquisition stopped immediately after an error: {error}\n"
              f"Queued downloads cancelled; verified local files retained in {staging}. "
              "No automatic retries of a denied request will be made.", file=sys.stderr, flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not failed:
            pool.shutdown(wait=True)
    manifest = {**inventory, "files": sorted(results, key=lambda item: item["relative_path"]),
                "completed_utc": utc_now(), "complete": True,
                "local_images_directory": "images",
                "integrity": "Every file matches Drive's MD5 and byte size; local SHA-256 recorded"}
    # A completed manifest is written once, before publishing the directory.
    manifest_file = staging / "acquisition_manifest.json"
    if manifest_file.exists():
        raise RuntimeError("Partial directory already has a completed manifest; inspect before recovery")
    write_new_json(manifest_file, manifest)
    if args.dest.exists():
        raise RuntimeError("Destination appeared during acquisition; refusing overwrite")
    staging.rename(args.dest)
    print(f"Complete snapshot: {args.dest.resolve()}", flush=True)
    print(f"Raw image root: {(args.dest / 'images').resolve()}", flush=True)


if __name__ == "__main__":
    main()
