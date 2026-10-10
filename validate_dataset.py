"""Verify every label, artifact hash, class ID, and source split before training."""
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from generate_dataset import sha256,xyxy_to_xywh,intersects
from generate_mixed_dataset import cell_rectangle

def validate(root):
 root=Path(root);audit=json.loads((root/'audit.json').read_text());mapping=json.loads((root/'class_mapping.json').read_text());records=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()];hashes=set();sources=defaultdict(set);source_paths=defaultdict(set);pixels=defaultdict(set);classes=defaultdict(set);counts=Counter()
 layoutcounts=defaultdict(Counter);layoutclasses=defaultdict(lambda:defaultdict(set))
 for r in records:
  if sha256(root/r['image'])!=r['image_sha256'] or sha256(root/r['label'])!=r['label_sha256']:raise ValueError('Artifact checksum mismatch')
  if r['image_sha256'] in hashes:raise ValueError('Duplicate composite image')
  hashes.add(r['image_sha256']);counts[r['split']]+=1
  with Image.open(root/r['image']) as image:
   if image.size!=(audit['image_size'],audit['image_size']):raise ValueError('Unexpected composite dimensions')
  if audit.get('layouts'):
   layout=r.get('layout')
   if layout not in audit['layouts']:raise ValueError('Missing or invalid layout')
   layoutcounts[r['split']][layout]+=1
   side=r.get('grid_side')
   if layout=='random':
    if side is not None or not 2<=len(r['sources'])<=4:raise ValueError('Invalid random-placement composition')
   else:
    if side!=int(layout[0]) or len(r['sources'])!=side*side:raise ValueError('Grid must contain one positive source in every cell')
    if {s['cell_index'] for s in r['sources']}!=set(range(side*side)):raise ValueError('Duplicate or missing grid cell')
    for s in r['sources']:
     cell=cell_rectangle(side,s['cell_index'],audit['image_size']);b=s['patch_xyxy']
     if s['cell_xyxy']!=cell or not(cell[0]<=b[0]<b[2]<=cell[2] and cell[1]<=b[1]<b[3]<=cell[3]):raise ValueError('Patch outside assigned grid cell')
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
   if audit.get('layouts'):layoutclasses[r['split']][r['layout']].add(c)
  for j,s in enumerate(r['sources']):
   sources[r['split']].add(s['sha256'])
   source_paths[r['split']].add(s['path'])
   if s.get('pixel_sha256'):pixels[r['split']].add(s['pixel_sha256'])
   boxes=[t['xyxy'] for t in r['targets'] if t['source_path']==s['path']]
   if not boxes:raise ValueError('Background-only source or grid cell')
   rect=s['patch_xyxy']
   if any(not(rect[0]-1e-6<=b[0]<b[2]<=rect[2]+1e-6 and rect[1]-1e-6<=b[1]<b[3]<=rect[3]+1e-6) for b in boxes):raise ValueError('Target outside its transformed source patch')
   for other in r['sources'][:j]:
    if intersects(s['patch_xyxy'],other['patch_xyxy']):raise ValueError('Overlapping contextual crops')
 for split in ['train','val','test']:
  if counts[split]!=audit['counts'][split]['images'] or classes[split]!=set(range(73)):raise ValueError('Split counts or class coverage mismatch')
  if audit.get('layouts'):
   for layout in audit['layouts']:
    if layoutcounts[split][layout]!=audit['counts'][split]['layouts'][layout]['images'] or layoutclasses[split][layout]!=set(range(73)):raise ValueError('Layout counts or class coverage mismatch')
  if audit.get('source_coverage_policy') and len(source_paths[split])!=audit['counts'][split]['unique_source_photos']:raise ValueError('Incomplete source-photo coverage')
  if audit.get('source_coverage_policy'):
   eligible=json.loads((root/'eligible_sources.json').read_text())
   if source_paths[split]!=set(eligible[split]):raise ValueError('Used source paths differ from the eligible annotation pool')
 for a,b in [('train','val'),('train','test'),('val','test')]:
  if sources[a]&sources[b] or source_paths[a]&source_paths[b] or pixels[a]&pixels[b]:raise ValueError('Source leakage')
 if len(records)<100:raise ValueError('Milestone requires at least 100 images')
 report={'passed':True,'images_verified':len(records),'split_counts':dict(counts),'classes_per_split':{k:len(v) for k,v in classes.items()},'target_instances':sum(len(r['targets']) for r in records),'checks':['all JPEG/label checksums','no duplicate composite bytes','official IDs and labels','normalized finite YOLO boxes','coordinate transform consistency','non-overlapping source crops','source hashes disjoint','73-class coverage in every split']}
 if audit.get('layouts'):
  report['layout_counts']={s:dict(c) for s,c in layoutcounts.items()};report['checks']+=['fully occupied 2x2/3x3 grids','patches and targets contained within cells','no background-only sources','73-class coverage in every split/layout']
 (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);a=p.parse_args();print(json.dumps(validate(a.dataset),indent=2))
