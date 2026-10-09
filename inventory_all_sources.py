"""Inventory every finalized photo without altering raw data or existing splits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.mpo', '.png', '.bmp', '.webp', '.tif', '.tiff', '.gif'}


def build(raw, prepared, output):
    mapping = json.loads((HERE / 'class_mapping.json').read_text())
    classes = {m['folder']: m for m in mapping}
    canonical = {}
    for split in ('train', 'val', 'test'):
        with (prepared / (split + '.csv')).open() as f:
            for row in csv.DictReader(f):
                canonical[row['path']] = {**row, 'split': split}
    with (prepared / 'exclusions.csv').open() as f:
        duplicates = {r['path']: r['duplicate_of'] for r in csv.DictReader(f) if r['reason'] == 'exact_file_duplicate'}
    files = sorted(p for p in raw.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    rows = []
    for p in files:
        relative = p.relative_to(raw).as_posix()
        original = duplicates.get(relative, relative)
        if original not in canonical:
            raise ValueError(f'Photo absent from finalized manifests: {relative}')
        row = canonical[original]
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != row['sha256']:
            raise ValueError(f'Finalized source checksum mismatch: {relative}')
        if classes[p.parent.name]['object_id'] != row['object_id']:
            raise ValueError('Object ID folder mismatch')
        rows.append({**row, 'path': relative, 'class_index': int(row['class_index']),
                     'raw_id': len(rows), 'duplicate_of': original if original != relative else None,
                     'background_filename_hint': '_bg_' in p.name.lower()})
    assert len(rows) == 7343 and len(canonical) == 7315 and len(duplicates) == 28
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'schema_version': 1, 'seed': 42, 'raw_photos': len(rows),
                                 'unique_photos': len(canonical), 'duplicate_copies': len(duplicates),
                                 'split_policy': 'Existing finalized seed-42 exact-deduplicated Milestone 1 source manifests',
                                 'raw_sources_modified': False, 'records': rows}, indent=2) + '\n')
    print(json.dumps({'raw_photos': len(rows), 'unique_photos': len(canonical), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, default=HERE / 'annotations/all_source_inventory.json')
    a = p.parse_args()
    build(a.raw, a.prepared, a.output)
