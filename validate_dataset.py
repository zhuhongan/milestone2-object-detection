"""Verify every label, artifact hash, class ID, and source split before training."""
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
from generate_dataset import sha256,xyxy_to_xywh,intersects

def validate(root):
 root=Path(root);audit=json.loads((root/'audit.json').read_text());mapping=json.loads((root/'class_mapping.json').read_text());records=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()];hashes=set();sources=defaultdict(set);classes=defaultdict(set);counts=Counter()
 for r in records:
  if sha256(root/r['image'])!=r['image_sha256'] or sha256(root/r['label'])!=r['label_sha256']:raise ValueError('Artifact checksum mismatch')
  if r['image_sha256'] in hashes:raise ValueError('Duplicate composite image')
  hashes.add(r['image_sha256']);counts[r['split']]+=1
  lines=(root/r['label']).read_text().splitlines()
  if len(lines)!=len(r['targets']) or len(lines)<2:raise ValueError('Multi-object label count mismatch')
  for line,t in zip(lines,r['targets']):
   values=line.split();c=int(values[0]);b=[float(x) for x in values[1:]]
   if len(b)!=4 or not all(math.isfinite(x) and 0<=x<=1 for x in b) or min(b[2:])<=0:raise ValueError('Invalid YOLO coordinates')
   m=mapping[c]
   if (c,m['object_id'],m['object_name'])!=(t['class_index'],t['object_id'],t['object_name']):raise ValueError('Official Object ID mismatch')
   expected=xyxy_to_xywh(t['xyxy'],audit['image_size'])
   if any(abs(a-b)>1e-7 for a,b in zip(expected,b)):raise ValueError('Transformed annotation mismatch')
   classes[r['split']].add(c)
  for j,s in enumerate(r['sources']):
   sources[r['split']].add(s['sha256'])
   for other in r['sources'][:j]:
    if intersects(s['patch_xyxy'],other['patch_xyxy']):raise ValueError('Overlapping contextual crops')
 for split in ['train','val','test']:
  if counts[split]!=audit['counts'][split]['images'] or classes[split]!=set(range(73)):raise ValueError('Split counts or class coverage mismatch')
 for a,b in [('train','val'),('train','test'),('val','test')]:
  if sources[a]&sources[b]:raise ValueError('Source leakage')
 if len(records)<100:raise ValueError('Milestone requires at least 100 images')
 report={'passed':True,'images_verified':len(records),'split_counts':dict(counts),'classes_per_split':{k:len(v) for k,v in classes.items()},'target_instances':sum(len(r['targets']) for r in records),'checks':['all JPEG/label checksums','no duplicate composite bytes','official IDs and labels','normalized finite YOLO boxes','coordinate transform consistency','non-overlapping source crops','source hashes disjoint','73-class coverage in every split']}
 (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);a=p.parse_args();print(json.dumps(validate(a.dataset),indent=2))
