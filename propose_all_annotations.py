"""Grounding DINO proposals for every unique source, not final reviewed labels."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/content/milestone2/annotation_job'))
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    import torch
    import transformers
    from PIL import Image, ImageOps
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from torchvision.ops import nms
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = a.root.resolve()
    output = root / 'output'
    output.mkdir(exist_ok=True)
    inventory = json.loads((root / 'all_source_inventory.json').read_text())
    prompts = json.loads((root / 'annotation_prompts.json').read_text())
    sources = [r for r in inventory['records'] if not r['duplicate_of']]
    if a.limit:
        # First smoke check covers all 73 classes rather than the first folder.
        sources = [next(r for r in sources if r['object_id'] == obj) for obj in sorted(prompts)][:a.limit]
    model_id = 'IDEA-Research/grounding-dino-base'
    revision_file = output / 'model_revision.json'
    revision = json.loads(revision_file.read_text())['revision'] if revision_file.exists() else HfApi().model_info(model_id).sha
    revision_file.write_text(json.dumps({'model': model_id, 'revision': revision}, indent=2) + '\n')
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=revision).to('cuda').eval()
    # Default model preprocessing is retained (short side 800, long side at most 1333).
    record_path = output / ('smoke_proposals.jsonl' if a.limit else 'proposals.jsonl')
    done = {r['path'] for r in map(json.loads, record_path.read_text().splitlines())} if record_path.exists() else set()
    todo = [r for r in sources if r['path'] not in done]
    started = time.monotonic()
    env = {'python': platform.python_version(), 'transformers': transformers.__version__,
           'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
           'model': model_id, 'revision': revision, 'seed': 42,
           'box_threshold': .15, 'text_threshold': .15, 'nms_iou': .5,
           'proposal_policy': 'All qualifying candidates retained for visual review; no confident ground-truth claim',
           'processor': processor.image_processor.to_dict(),
           'inventory_sha256': hashlib.sha256((root / 'all_source_inventory.json').read_bytes()).hexdigest(),
           'prompts_sha256': hashlib.sha256((root / 'annotation_prompts.json').read_bytes()).hexdigest(),
           'packages': subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    (output / 'environment.json').write_text(json.dumps(env, indent=2) + '\n')
    with record_path.open('a', buffering=1) as f:
        for start in range(0, len(todo), a.batch):
            batch = todo[start:start+a.batch]
            images = []
            for row in batch:
                path = root / 'images' / row['path']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
                with Image.open(path) as im:
                    im.seek(0)
                    images.append(ImageOps.exif_transpose(im).convert('RGB'))
            inputs = processor(images=images, text=[prompts[r['object_id']] for r in batch],
                               padding=True, return_tensors='pt').to('cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=.15, text_threshold=.15,
                target_sizes=[(im.height, im.width) for im in images])
            for row, im, result in zip(batch, images, results):
                boxes = result['boxes'].float().cpu()
                scores = result['scores'].float().cpu()
                if len(boxes):
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, im.width)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, im.height)
                    keep = nms(boxes, scores, .5).tolist()
                else:
                    keep = []
                candidates = []
                for i in keep:
                    b = boxes[i].tolist()
                    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                        continue
                    candidates.append({'score': float(scores[i]), 'phrase': result['text_labels'][i],
                                       'xyxy_normalized': [b[0]/im.width, b[1]/im.height, b[2]/im.width, b[3]/im.height]})
                f.write(json.dumps({**row, 'oriented_width': im.width, 'oriented_height': im.height,
                                    'prompt': prompts[row['object_id']], 'candidates': candidates,
                                    'status': 'proposal_pending_visual_review'}) + '\n')
            completed = len(done) + min(start + a.batch, len(todo))
            if start % 80 == 0 or completed == len(sources):
                print(json.dumps({'proposals': completed, 'total': len(sources),
                                  'seconds': round(time.monotonic()-started, 1)}), flush=True)
    records = [json.loads(line) for line in record_path.read_text().splitlines()]
    assert {r['path'] for r in records} == {r['path'] for r in sources}
    (output / ('smoke_completed.json' if a.limit else 'completed.json')).write_text(json.dumps({
        'completed': True, 'photos_processed': len(records), 'raw_photos': inventory['raw_photos'],
        'duplicate_copies': inventory['duplicate_copies'], 'proposals_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest(),
        'candidate_counts': dict(Counter(len(r['candidates']) for r in records)),
        'annotation_status': 'Proposals require visual review before use'}, indent=2) + '\n')
    print('All requested proposal records saved', flush=True)


if __name__ == '__main__':
    main()
