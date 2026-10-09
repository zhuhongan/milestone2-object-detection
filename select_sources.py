"""Create review sheets from a deterministic source-disjoint annotation subset."""
import csv,json,random
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/final-local/images'
MAPPING=json.loads((ROOT/'data/prepared-local-v2/class_mapping.json').read_text())
pools={}
for split in ['train','val','test']:
 rows=list(csv.DictReader((ROOT/f'data/prepared-local-v2/{split}.csv').open()))
 pools[split]={m['object_id']:[r for r in rows if r['object_id']==m['object_id'] and '_bg_' not in r['path'].lower()] for m in MAPPING}
records=[];rng=random.Random(42)
for m in MAPPING:
 for split,count in [('train',3),('val',1),('test',1)]:
  candidates=pools[split][m['object_id']].copy();rng.shuffle(candidates)
  for r in candidates[:count]:
   records.append({**r,'split':split,'review_id':len(records),'bbox_xyxy_224':None,'status':'pending'})
(ROOT/'milestone2/annotations/source_review.json').write_text(json.dumps(records,indent=2)+'\n')
for start in range(0,len(MAPPING),7):
 group=records[start*5:min(start+7,len(MAPPING))*5]
 sheet=Image.new('RGB',(1200, len(group)//5*265),'white');draw=ImageDraw.Draw(sheet)
 for k,r in enumerate(group):
  x=(k%5)*240;y=(k//5)*265
  with Image.open(RAW/r['path']) as image:
   image.seek(0);im=ImageOps.exif_transpose(image).convert('RGB').resize((224,224))
  sheet.paste(im,(x+8,y+30));draw.text((x+8,y+2),f"#{r['review_id']} {r['object_id']} {r['split']}",fill='black')
  draw.text((x+8,y+16),r['object_name'][:31],fill='black')
 sheet.save(ROOT/f'tmp/milestone2/annotation_sheets/classes_{start+1:02d}_{min(start+7,len(MAPPING)):02d}.png')
print('Review sources',len(records))
