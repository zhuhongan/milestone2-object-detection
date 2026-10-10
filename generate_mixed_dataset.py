"""Reproducible 2x2 / 3x3 / random-placement composites from immutable sources."""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from generate_dataset import intersects, oriented_rgb, sha256, transformed_boxes, xyxy_to_xywh

HERE = Path(__file__).resolve().parent
LAYOUTS = ('2x2', '3x3', 'random')


def layout_counts(count):
    return {'2x2': count * 2 // 5, '3x3': count * 2 // 5,
            'random': count - 2 * (count * 2 // 5)}


def cell_rectangle(side, index, size):
    row, col = divmod(index, side)
    return [col * size // side, row * size // side,
            (col + 1) * size // side, (row + 1) * size // side]


def grid_geometry(width, height, cell, rng):
    left, top, right, bottom = cell
    # Full photographs are letterboxed, never stretched or cropped in grid cells.
    margin = rng.uniform(.02, .10)
    available_w = (right - left) * (1 - 2 * margin)
    available_h = (bottom - top) * (1 - 2 * margin)
    factor = min(available_w / width, available_h / height)
    pw, ph = max(2, round(width * factor)), max(2, round(height * factor))
    ox = rng.randint(left + 4, right - pw - 4)
    oy = rng.randint(top + 4, bottom - ph - 4)
    return pw, ph, ox, oy


def generate(raw_root, annotations, classes, output, seed=42, counts=(8000, 1000, 1000), size=640):
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    if output == raw_root or raw_root in output.parents:
        raise ValueError('Output must be separate from locked raw images')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing dataset')
    if len(counts) != 3 or any(c <= 0 for c in counts):
        raise ValueError('Provide three positive split counts')
    mapping = json.loads(Path(classes).read_text())
    if len(mapping) != 73 or {m['class_index'] for m in mapping} != set(range(73)):
        raise ValueError('Expected the final 73-class mapping')
    records = json.loads(Path(annotations).read_text())['selected_sources']
    pools = {s: defaultdict(list) for s in ('train', 'val', 'test')}
    paths = {s: set() for s in pools}
    hashes = {s: set() for s in pools}
    pixels = {s: set() for s in pools}
    cache = {}
    for source in records:
        c, split = int(source['class_index']), source['split']
        if (source['object_id'], source['object_name']) != (mapping[c]['object_id'], mapping[c]['object_name']):
            raise ValueError('Final Object ID mismatch')
        if sha256(raw_root / source['path']) != source['sha256']:
            raise ValueError('Locked source checksum mismatch')
        image = oriented_rgb(raw_root / source['path'])
        w, h = image.size
        boxes = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in source['boxes_xyxy_normalized']]
        if not boxes or any(not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h) for b in boxes):
            raise ValueError('Every selected source must have valid visible-target boxes')
        cache[source['path']] = image
        pools[split][c].append({**source, 'class_index': c, 'native_boxes': boxes})
        paths[split].add(source['path'])
        hashes[split].add(source['sha256'])
        pixels[split].add(source['pixel_sha256'])
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if paths[a] & paths[b] or hashes[a] & hashes[b] or pixels[a] & pixels[b]:
            raise ValueError('Source leakage across splits')
    for split, count in zip(pools, counts):
        if set(pools[split]) != set(range(73)):
            raise ValueError('Missing source class')
        budgets = layout_counts(count)
        if min(budgets.values()) < 73:
            raise ValueError('Each split/layout requires at least 73 images for class coverage')
        if sum(n // 73 for n in budgets.values()) < max(map(len, pools[split].values())):
            raise ValueError('Insufficient budget to guarantee every positive source is used')
    output.mkdir(parents=True, exist_ok=True)
    manifest, audit = [], {}
    for split, count in zip(pools, counts):
        split_seed = seed + {'train': 0, 'val': 100000, 'test': 200000}[split]
        rng, np_rng = random.Random(split_seed), np.random.default_rng(split_seed)
        budgets = layout_counts(count)
        layouts = [layout for layout, number in budgets.items() for _ in range(number)]
        rng.shuffle(layouts)
        schedules, indices, positions = {}, Counter(), Counter()
        cycles = {c: pool.copy() for c, pool in pools[split].items()}
        for pool in cycles.values():
            rng.shuffle(pool)
        for layout, number in budgets.items():
            schedules[layout] = []
            while len(schedules[layout]) < number:
                cycle = list(range(73))
                rng.shuffle(cycle)
                schedules[layout].extend(cycle)
        used, classcounts, objectcounts = set(), Counter(), Counter()
        by_layout = {layout: {'images': 0, 'target_instances': 0, 'per_class_instances': Counter(),
                              'objects_per_image': Counter()} for layout in LAYOUTS}
        for leaf in ('images', 'labels'):
            (output / leaf / split).mkdir(parents=True, exist_ok=True)
        for index, layout in enumerate(layouts):
            side = int(layout[0]) if layout != 'random' else None
            number = side * side if side else rng.randint(2, 4)
            anchor = schedules[layout][indices[layout]]
            indices[layout] += 1
            selected = [anchor] + rng.sample([c for c in range(73) if c != anchor], number - 1)
            rng.shuffle(selected)
            base = np.array([rng.randint(185, 240) for _ in range(3)], dtype=np.float32)
            canvas = Image.fromarray(np.uint8(np.clip(base + np_rng.normal(0, 2, (size, size, 1)), 0, 255)), 'RGB')
            sources, targets, placed = [], [], []
            for cell_index, c in enumerate(selected):
                source = cycles[c][positions[c] % len(cycles[c])]
                positions[c] += 1
                used.add(source['path'])
                image = cache[source['path']]
                w, h = image.size
                boxes = source['native_boxes']
                if side:
                    cell = cell_rectangle(side, cell_index, size)
                    crop = [0, 0, w, h]
                    pw, ph, ox, oy = grid_geometry(w, h, cell, rng)
                else:
                    cell = None
                    l, t = min(b[0] for b in boxes), min(b[1] for b in boxes)
                    r, b = max(b[2] for b in boxes), max(b[3] for b in boxes)
                    context = rng.uniform(.05, .15)
                    crop = [max(0, math.floor(l - (r-l)*context)), max(0, math.floor(t - (b-t)*context)),
                            min(w, math.ceil(r + (r-l)*context)), min(h, math.ceil(b + (b-t)*context))]
                    factor = rng.randint(150, 235) / max(crop[2]-crop[0], crop[3]-crop[1])
                    pw, ph = max(2, round((crop[2]-crop[0])*factor)), max(2, round((crop[3]-crop[1])*factor))
                    found = False
                    for _ in range(6):
                        for _ in range(1000):
                            ox, oy = rng.randint(8, size-pw-8), rng.randint(8, size-ph-8)
                            if all(not intersects([ox, oy, ox+pw, oy+ph], other) for other in placed):
                                found = True
                                break
                        if found:
                            break
                        pw, ph = max(2, round(pw*.8)), max(2, round(ph*.8))
                    if not found:
                        raise RuntimeError('Could not place random crops')
                patch = image.crop(tuple(crop))
                flip = split == 'train' and rng.random() < .5
                if flip:
                    patch = ImageOps.mirror(patch)
                if split == 'train':
                    patch = ImageEnhance.Brightness(patch).enhance(rng.uniform(.9, 1.1))
                    patch = ImageEnhance.Contrast(patch).enhance(rng.uniform(.9, 1.1))
                patch = patch.resize((pw, ph), Image.Resampling.BILINEAR)
                canvas.paste(patch, (ox, oy))
                rectangle = [ox, oy, ox+pw, oy+ph]
                placed.append(rectangle)
                moved = transformed_boxes(boxes, crop, (pw/(crop[2]-crop[0]), ph/(crop[3]-crop[1])), (ox, oy), flip)
                for box in moved:
                    targets.append({'class_index': c, 'object_id': source['object_id'],
                                    'object_name': source['object_name'], 'xyxy': box, 'source_path': source['path']})
                    classcounts[c] += 1
                    by_layout[layout]['per_class_instances'][source['object_id']] += 1
                sources.append({'path': source['path'], 'sha256': source['sha256'], 'pixel_sha256': source['pixel_sha256'],
                                'crop_xyxy': crop, 'patch_xyxy': rectangle, 'flipped': flip,
                                'cell_index': cell_index if side else None, 'cell_xyxy': cell})
            stem = f'{split}_{index:05d}'
            imagepath, labelpath = output/f'images/{split}/{stem}.jpg', output/f'labels/{split}/{stem}.txt'
            canvas.save(imagepath, quality=95, subsampling=0)
            labelpath.write_text(''.join(f"{t['class_index']} " + ' '.join(f'{v:.8f}' for v in xyxy_to_xywh(t['xyxy'], size))+'\n' for t in targets))
            manifest.append({'image': imagepath.relative_to(output).as_posix(), 'image_sha256': sha256(imagepath),
                             'label': labelpath.relative_to(output).as_posix(), 'label_sha256': sha256(labelpath),
                             'split': split, 'layout': layout, 'grid_side': side, 'sources': sources, 'targets': targets})
            objectcounts[len(targets)] += 1
            by_layout[layout]['images'] += 1
            by_layout[layout]['target_instances'] += len(targets)
            by_layout[layout]['objects_per_image'][len(targets)] += 1
            if index % 1000 == 0:
                print(f'{split}: generated {index+1}/{count}', flush=True)
        if used != paths[split]:
            raise ValueError('Not every eligible positive source was used')
        audit[split] = {'images': count, 'target_instances': sum(classcounts.values()), 'unique_source_photos': len(used),
                        'actually_used_source_photos': len(used), 'objects_per_image': dict(sorted(objectcounts.items())),
                        'per_class_instances': {mapping[c]['object_id']: classcounts[c] for c in range(73)},
                        'layouts': by_layout}
    (output/'data.yaml').write_text('path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f"  {m['class_index']}: {m['object_id']}\n" for m in mapping))
    (output/'class_mapping.json').write_text(json.dumps(mapping, indent=2)+'\n')
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (output/'eligible_sources.json').write_text(json.dumps({s: sorted(p) for s, p in paths.items()}, indent=2)+'\n')
    record = {'schema_version': 2, 'seed': seed, 'image_size': size, 'counts': audit,
              'source_annotations_sha256': sha256(annotations), 'class_mapping_sha256': sha256(classes),
              'source_disjoint': True, 'raw_sources_modified': False, 'layouts': list(LAYOUTS),
              'layout_proportions': {'2x2': .4, '3x3': .4, 'random': .2},
              'source_coverage_policy': 'Seeded source cycles; every eligible positive source used at least once',
              'composition': '2x2 and 3x3 full-photo letterboxed grids plus 2-4 randomly placed contextual crops; all cells positive; multiple target instances retained',
              'limitations': 'Only assigned folder objects annotated; incidental classes not exhaustively labelled; near-capture independence and real-scene generalization not established'}
    (output/'audit.json').write_text(json.dumps(record, indent=2)+'\n')
    preview = Image.new('RGB', (size*3, size*2), 'white')
    for j, row in enumerate([next(r for r in manifest if r['split']==split and r['layout']==layout)
                             for split in ('train', 'test') for layout in LAYOUTS]):
        image = Image.open(output/row['image']).convert('RGB')
        draw = ImageDraw.Draw(image)
        for target in row['targets']:
            draw.rectangle(target['xyxy'], outline='red', width=3)
            draw.text((target['xyxy'][0], target['xyxy'][1]), target['object_id'], fill='red')
        preview.paste(image, (j%3*size, j//3*size))
    preview.save(output/'generation_preview.jpg')
    print(json.dumps({s: {'images': r['images'], 'instances': r['target_instances'], 'sources': r['unique_source_photos'],
                          'layouts': {k: v['images'] for k, v in r['layouts'].items()}} for s, r in audit.items()}, indent=2))
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--annotations', type=Path, default=HERE/'annotations/all_photos_seed42/positive_source_annotations.json')
    p.add_argument('--classes', type=Path, default=HERE/'class_mapping.json')
    p.add_argument('--output', type=Path, default=HERE/'data/mixed_layouts_seed42')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--counts', type=int, nargs=3, default=(8000, 1000, 1000))
    a = p.parse_args()
    generate(a.raw_root, a.annotations, a.classes, a.output, a.seed, a.counts)
