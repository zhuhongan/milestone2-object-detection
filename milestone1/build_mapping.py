"""Extract official class identities from a read-only Sheet CSV snapshot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

SHEET_URL = "https://docs.google.com/spreadsheets/d/1OCBWlTK8JSzrgUxON3thJWAKRKU2Pi-V9Ssn2BbDxds/edit#gid=0"


def build(source, output):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [{k.strip(): v.strip() for k, v in row.items() if k} for row in reader]
    mapping = []
    ids, folders = set(), set()
    for row in rows:
        if not any(row.values()):
            continue
        oid, folder, name = (row.get(k, "") for k in ("OBJECT_ID", "OBJECT_FOLDER", "OBJECT_NAME"))
        if not oid or not folder or not name or oid in ids or folder in folders:
            raise ValueError("Missing or duplicate official ID, folder, or object name")
        ids.add(oid)
        folders.add(folder)
        mapping.append({"folder": folder, "object_id": oid, "object_name": name})
    if not mapping:
        raise ValueError("The Sheet contains no mappings")
    mapping.sort(key=lambda row: row["object_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["folder", "object_id", "object_name"])
        writer.writeheader()
        writer.writerows(mapping)
    provenance = {"source_url": SHEET_URL, "sheet_gid": "0", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "mapping_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "class_count": len(mapping),
                  "transform": "Trim outer whitespace only; preserve official IDs, names, and folders; sort by object_id.",
                  "verified_against_sheet": True}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"classes": len(mapping), "output": str(output), "unique_ids": len(ids), "unique_folders": len(folders)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("metadata/object_ids_sheet.csv"))
    parser.add_argument("--output", type=Path, default=Path("metadata/object_mapping.csv"))
    args = parser.parse_args()
    build(args.source, args.output)
