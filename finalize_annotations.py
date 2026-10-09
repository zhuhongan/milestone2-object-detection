"""Finalize visually reviewed boxes, preserving the raw source files."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]

def main():
 records=json.loads((ROOT/'milestone2/annotations/source_review.json').read_text())
 boxes=json.loads((ROOT/'milestone2/annotations/reviewed_boxes.json').read_text())
 candidates=json.loads((ROOT/'milestone2/annotations/replacement_candidates.json').read_text())
 replacements={15:(365,[102,99,156,144]),19:(369,[9,112,187,187]),60:(372,[56,56,202,208]),77:(375,[88,35,147,134]),122:(379,[66,62,148,197]),215:(380,[78,86,139,160]),294:(385,[60,142,141,214]),304:(386,[54,67,155,190]),309:(389,[10,87,208,149]),336:(393,[0,21,224,224])}
 exclusions=[];final=[]
 for r in records:
  value=boxes[str(r['review_id'])]
  if value is None:
   exclusions.append({'path':r['path'],'object_id':r['object_id'],'split':r['split'],'reason':'Target not visible or too small for a reliable training annotation','replacement_review_id':replacements[r['review_id']][0]})
   rid,value=replacements[r['review_id']];r=next(c for c in candidates if c['review_id']==rid).copy()
  value=value if isinstance(value[0],list) else [value]
  for b in value:assert 0<=b[0]<b[2]<=224 and 0<=b[1]<b[3]<=224,(r['review_id'],b)
  raw=ROOT/'data/raw/final-local/images'/r['path'];assert hashlib.sha256(raw.read_bytes()).hexdigest()==r['sha256']
  r={k:r[k] for k in ['review_id','path','object_id','object_name','class_index','split','sha256','pixel_sha256']}
  r['boxes_xyxy_normalized']=[[a/224 for a in b] for b in value]
  r['review_method']='Visual annotation by assistant on EXIF-oriented primary RGB frame'
  final.append(r)
 assert len(final)==365
 assert len({r['path'] for r in final})==365
 for split in ['train','val','test']:
  c=Counter(r['object_id'] for r in final if r['split']==split);assert len(c)==73;assert set(c.values())==({3} if split=='train' else {1})
 out=ROOT/'milestone2/annotations'
 (out/'source_annotations.json').write_text(json.dumps({'schema_version':1,'seed':42,'coordinate_system':'EXIF-oriented primary RGB frame; normalized xyxy','selected_sources':final,'reviewer':'assistant visual annotation; no independent human verification','selection':'Seed-42 subset from locked source split manifests; 3 training, 1 validation, 1 test photo per official class. Filename-tagged background photos not eligible; unsuitable candidates replaced within the same source split.','limitations':'Bounding boxes mark visible targets, including partial views; source photographs and contextual backgrounds may be correlated across source splits. Pair of sneakers annotated as two instances.'},indent=2)+'\n')
 (out/'annotation_exclusions.json').write_text(json.dumps(exclusions,indent=2)+'\n')
 # Annotated originals provide a reproducible review record.
 byid={int(r['object_id'][3:]):[] for r in final}
 for r in final:byid[int(r['object_id'][3:])].append(r)
 for start in range(1,74,7):
  group=[r for c in range(start,min(start+7,74)) for r in byid[c]]
  sheet=Image.new('RGB',(1200,265*((len(group)+4)//5)),'white');draw=ImageDraw.Draw(sheet)
  for k,r in enumerate(group):
   x=k%5*240;y=k//5*265
   with Image.open(ROOT/'data/raw/final-local/images'/r['path']) as image:
    image.seek(0);im=ImageOps.exif_transpose(image).convert('RGB').resize((224,224))
   dr=ImageDraw.Draw(im)
   for b in r['boxes_xyxy_normalized']:dr.rectangle([a*224 for a in b],outline='red',width=2)
   sheet.paste(im,(x+8,y+30));draw.text((x+8,y+2),f"#{r['review_id']} {r['object_id']} {r['split']}",fill='black');draw.text((x+8,y+16),r['object_name'][:31],fill='black')
  sheet.save(ROOT/f'tmp/milestone2/annotation_sheets/reviewed_{start:02d}.png')
 print('Finalized',len(final),'source photos,',sum(len(r['boxes_xyxy_normalized']) for r in final),'target boxes')
if __name__=='__main__':main()
