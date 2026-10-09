"""Generate source-disjoint YOLO composites from reviewed object boxes."""
import argparse,csv,hashlib,json,math,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageEnhance,ImageDraw

HERE=Path(__file__).resolve().parent

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def oriented_rgb(path):
 with Image.open(path) as image:
  image.seek(0);return ImageOps.exif_transpose(image).convert('RGB')
def intersects(a,b,gap=8):return not(a[2]+gap<=b[0] or b[2]+gap<=a[0] or a[3]+gap<=b[1] or b[3]+gap<=a[1])
def xyxy_to_xywh(b,size):
 x1,y1,x2,y2=b;return [(x1+x2)/(2*size),(y1+y2)/(2*size),(x2-x1)/size,(y2-y1)/size]
def transformed_boxes(boxes,crop,scale,offset,flip=False):
 l,t,r,b=crop;sx,sy=scale;ox,oy=offset;out=[]
 for x1,y1,x2,y2 in boxes:
  if flip:x1,x2=r-x2,r-x1
  else:x1,x2=x1-l,x2-l
  out.append([ox+x1*sx,oy+(y1-t)*sy,ox+x2*sx,oy+(y2-t)*sy])
 return out

def generate(raw_root,annotations,classes,output,seed=42,counts=(1168,146,146),size=640):
 raw_root=Path(raw_root).resolve();output=Path(output).resolve()
 if output==raw_root or raw_root in output.parents:raise ValueError('Output must be separate from locked raw images')
 if output.exists() and any(output.iterdir()):raise FileExistsError('Refusing to overwrite an existing generated dataset')
 mapping=json.loads(Path(classes).read_text());records=json.loads(Path(annotations).read_text())['selected_sources']
 assert len(mapping)==73 and len({m['object_id'] for m in mapping})==73
 pools={s:defaultdict(list) for s in ['train','val','test']}
 source_sets={s:set() for s in pools};hash_sets={s:set() for s in pools};pixels={s:set() for s in pools}
 cache={}
 for r in records:
  m=mapping[int(r['class_index'])]
  if (m['object_id'],m['object_name'])!=(r['object_id'],r['object_name']):raise ValueError('Official class mapping mismatch')
  path=raw_root/r['path']
  if sha256(path)!=r['sha256']:raise ValueError(f'Source checksum mismatch: {r["path"]}')
  im=oriented_rgb(path);w,h=im.size;cache[r['path']]=im
  boxes=[[b[0]*w,b[1]*h,b[2]*w,b[3]*h] for b in r['boxes_xyxy_normalized']]
  for b in boxes:
   if not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError('Invalid source box')
  r={**r,'class_index':int(r['class_index']),'native_boxes':boxes};pools[r['split']][r['class_index']].append(r)
  source_sets[r['split']].add(r['path']);hash_sets[r['split']].add(r['sha256']);pixels[r['split']].add(r['pixel_sha256'])
 for a,b in [('train','val'),('train','test'),('val','test')]:
  if source_sets[a]&source_sets[b] or hash_sets[a]&hash_sets[b] or pixels[a]&pixels[b]:raise ValueError('Source leakage across splits')
 for split in pools:
  if set(pools[split])!=set(range(73)):raise ValueError(f'Missing class in {split}')
 output.mkdir(parents=True,exist_ok=True);manifest=[];audit={};np_rng=np.random.default_rng(seed)
 for split,count in zip(pools,counts):
  rng=random.Random(seed+{'train':0,'val':100000,'test':200000}[split]);schedule=[]
  while len(schedule)<count:
   cycle=list(range(73));rng.shuffle(cycle);schedule.extend(cycle)
  for leaf in ['images','labels']:(output/leaf/split).mkdir(parents=True,exist_ok=True)
  classcounts=Counter();objectcounts=Counter()
  for i in range(count):
   n=rng.randint(2,4);selected=[schedule[i]]+rng.sample([c for c in range(73) if c!=schedule[i]],n-1)
   base=np.array([rng.randint(185,240) for _ in range(3)],dtype=np.float32)
   noise=np_rng.normal(0,2,(size,size,1));canvas=Image.fromarray(np.uint8(np.clip(base+noise,0,255)),'RGB')
   placements=[];targets=[];sources=[]
   for c in selected:
    r=rng.choice(pools[split][c]);im=cache[r['path']];w,h=im.size;boxes=r['native_boxes']
    x1=min(b[0] for b in boxes);y1=min(b[1] for b in boxes);x2=max(b[2] for b in boxes);y2=max(b[3] for b in boxes)
    # Tight rectangular object crop with 5-15% context. Background within the crop is retained.
    context=rng.uniform(.05,.15)
    crop=[max(0,math.floor(x1-(x2-x1)*context)),max(0,math.floor(y1-(y2-y1)*context)),min(w,math.ceil(x2+(x2-x1)*context)),min(h,math.ceil(y2+(y2-y1)*context))]
    patch=im.crop(tuple(crop));longside=rng.randint(150,235);factor=longside/max(patch.size)
    pw=max(2,round(patch.width*factor));ph=max(2,round(patch.height*factor))
    flip=split=='train' and rng.random()<.5
    if flip:patch=ImageOps.mirror(patch)
    if split=='train':
     patch=ImageEnhance.Brightness(patch).enhance(rng.uniform(.9,1.1));patch=ImageEnhance.Contrast(patch).enhance(rng.uniform(.9,1.1))
    patch=patch.resize((pw,ph),Image.Resampling.BILINEAR)
    placed=False
    for shrink in range(6):
     for attempt in range(1000):
      ox=rng.randint(8,size-pw-8);oy=rng.randint(8,size-ph-8);rect=[ox,oy,ox+pw,oy+ph]
      if all(not intersects(rect,p) for p in placements):placed=True;break
     if placed:break
     pw=max(2,round(pw*.8));ph=max(2,round(ph*.8));patch=patch.resize((pw,ph),Image.Resampling.BILINEAR)
    if not placed:raise RuntimeError('No valid non-overlapping placement')
    canvas.paste(patch,(ox,oy));placements.append(rect)
    moved=transformed_boxes(boxes,crop,(pw/(crop[2]-crop[0]),ph/(crop[3]-crop[1])),(ox,oy),flip)
    for b in moved:targets.append({'class_index':c,'object_id':r['object_id'],'object_name':r['object_name'],'xyxy':b,'source_path':r['path']});classcounts[c]+=1
    sources.append({'path':r['path'],'sha256':r['sha256'],'crop_xyxy':crop,'patch_xyxy':rect,'flipped':flip})
   stem=f'{split}_{i:05d}';imagepath=output/f'images/{split}/{stem}.jpg';labelpath=output/f'labels/{split}/{stem}.txt'
   canvas.save(imagepath,quality=95,subsampling=0)
   labelpath.write_text(''.join(f"{t['class_index']} "+' '.join(f'{v:.8f}' for v in xyxy_to_xywh(t['xyxy'],size))+'\n' for t in targets))
   manifest.append({'image':imagepath.relative_to(output).as_posix(),'image_sha256':sha256(imagepath),'label':labelpath.relative_to(output).as_posix(),'label_sha256':sha256(labelpath),'split':split,'sources':sources,'targets':targets})
   objectcounts[len(targets)]+=1
  audit[split]={'images':count,'target_instances':sum(classcounts.values()),'unique_source_photos':len(source_sets[split]),'per_class_instances':{mapping[c]['object_id']:classcounts[c] for c in range(73)},'objects_per_image':dict(sorted(objectcounts.items()))}
 yaml='path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f"  {m['class_index']}: {m['object_id']}\n" for m in mapping)
 (output/'data.yaml').write_text(yaml);(output/'class_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
 (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
 (output/'audit.json').write_text(json.dumps({'schema_version':1,'seed':seed,'image_size':size,'counts':audit,'source_annotations_sha256':sha256(annotations),'class_mapping_sha256':sha256(classes),'source_disjoint':True,'raw_sources_modified':False,'composition':'2-4 distinct selected classes; non-overlapping rectangular contextual crops; multiple instances retained; random scale and placement; seed-42 frozen manifests','limitations':'Synthetic rectangular patches retain within-crop background; only 3 train and 1 val/test source photo per class; near-duplicate capture independence not established.'},indent=2)+'\n')
 preview=Image.new('RGB',(size*3,size*2),'white')
 for j,r in enumerate([r for r in manifest if r['split']=='train'][:6]):
  im=Image.open(output/r['image']).convert('RGB');draw=ImageDraw.Draw(im)
  for t in r['targets']:draw.rectangle(t['xyxy'],outline='red',width=3);draw.text((t['xyxy'][0],t['xyxy'][1]),t['object_id'],fill='red')
  preview.paste(im,(j%3*size,j//3*size))
 preview.save(output/'generation_preview.jpg');print(json.dumps(audit,indent=2))
 return audit

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--annotations',type=Path,default=HERE/'annotations/source_annotations.json');p.add_argument('--classes',type=Path,default=HERE/'class_mapping.json');p.add_argument('--output',type=Path,default=HERE/'data/synthetic_seed42');p.add_argument('--seed',type=int,default=42)
 a=p.parse_args();generate(a.raw_root,a.annotations,a.classes,a.output,a.seed)
