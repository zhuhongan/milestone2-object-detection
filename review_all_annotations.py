"""Build exhaustive box review sheets and apply explicit per-photo review decisions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
from evaluate_predictions import iou

HERE = Path(__file__).resolve().parent


def font(size):
    try:
        return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)
    except OSError:
        return ImageFont.load_default()


def draft(row):
    candidates = row['candidates']
    if not candidates:
        return []
    selected = []
    for c in candidates:
        if selected and c['score'] < max(.25, candidates[0]['score'] * .72):
            continue
        b = c['xyxy_normalized']
        area = (b[2]-b[0])*(b[3]-b[1])
        if area < .002:
            continue
        contained = False
        for prior in selected:
            intersection = max(0, min(b[2], prior[2])-max(b[0], prior[0])) * max(0, min(b[3], prior[3])-max(b[1], prior[1]))
            if intersection / area > .85 or iou(b, prior) > .45:
                contained = True
        if not contained:
            selected.append(b)
    return selected


def prepare(raw, proposals, output, inventory, smoke=False):
    records = [json.loads(line) for line in proposals.read_text().splitlines()]
    prior = {r['path']: r for r in json.loads((HERE/'annotations/source_annotations.json').read_text())['selected_sources']}
    groups = defaultdict(list)
    for row in records:
        row['proposed_boxes_xyxy_normalized'] = prior[row['path']]['boxes_xyxy_normalized'] if row['path'] in prior else draft(row)
        row['initial_method'] = 'Existing visual annotation' if row['path'] in prior else 'Grounding DINO proposal awaiting visual review'
        groups[row['object_id']].append(row)
    output.mkdir(parents=True, exist_ok=True)
    (output/'draft_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    sheets=[]
    # Ten columns of native 224-pixel photographs; split each class into pages of 50.
    for obj, rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['path'])
        for page,start in enumerate(range(0,len(rows),50),1):
            subset=rows[start:start+50];columns=10;tilew=224;tileh=253
            sheet=Image.new('RGB',(columns*tilew,34+((len(subset)+columns-1)//columns)*tileh),'white');draw=ImageDraw.Draw(sheet)
            draw.text((5,4),f"{obj} {subset[0]['object_name']}  page {page}  red boxes are drafts for review",font=font(20),fill='black')
            for j,row in enumerate(subset):
                x=j%columns*tilew;y=34+j//columns*tileh
                with Image.open(raw/row['path']) as image:
                    image.seek(0);im=ImageOps.exif_transpose(image).convert('RGB').resize((224,224))
                d=ImageDraw.Draw(im)
                for k,b in enumerate(row['proposed_boxes_xyxy_normalized']):
                    box=[v*224 for v in b];d.rectangle(box,outline='red',width=2);d.text((box[0],max(0,box[1]-17)),str(k),font=font(16),fill='red')
                draw.text((x+3,y),f"#{row['raw_id']} {Path(row['path']).name}",font=font(13),fill='black')
                best=row['candidates'][0]['score'] if row['candidates'] else 0
                draw.text((x+3,y+14),f"{len(row['proposed_boxes_xyxy_normalized'])} box  score {best:.2f}",font=font(12),fill='black')
                sheet.paste(im,(x,y+29))
            name=f'{obj}_page{page:02d}.png';sheet.save(output/name)
            sheets.append({'sheet':name,'object_id':obj,'raw_ids':[r['raw_id'] for r in subset]})
    (output/'sheet_index.json').write_text(json.dumps(sheets,indent=2)+'\n')
    print(json.dumps({'records':len(records),'sheets':len(sheets),'output':str(output)},indent=2))


def finalize(raw, review, decisions, output, inventory):
    data=json.loads(inventory.read_text());rows=[json.loads(line) for line in (review/'draft_records.jsonl').read_text().splitlines()]
    decisions=json.loads(decisions.read_text());checked=set(decisions['reviewed_sheets']);sheets=json.loads((review/'sheet_index.json').read_text())
    if checked!={s['sheet'] for s in sheets}:raise ValueError('Every source review sheet must have an explicit review receipt')
    overrides=decisions.get('overrides',{});final={};counts=Counter()
    for r in rows:
        change=overrides.get(str(r['raw_id']),{})
        boxes=change.get('boxes_xyxy_normalized',r['proposed_boxes_xyxy_normalized'])
        status=change.get('status','target_present' if boxes else 'target_absent')
        if status not in {'target_present','target_absent'}:raise ValueError(f"Unresolved annotation {r['raw_id']}")
        if bool(boxes)!=(status=='target_present'):raise ValueError('Status and box count disagree')
        for b in boxes:
            if len(b)!=4 or not 0<=b[0]<b[2]<=1 or not 0<=b[1]<b[3]<=1:raise ValueError('Invalid reviewed box')
        final[r['path']]={**r,'boxes_xyxy_normalized':boxes,'status':status,
                         'review_method':'Grounding DINO assisted; every image visually reviewed by assistant on contact sheet; overrides recorded; no independent human review',
                         'review_note':change.get('reason','Contact sheet draft confirmed' if boxes else 'No visible target of the folder Object ID'),
                         'duplicate_of':None}
        counts[status]+=1
    all_rows=[]
    for item in data['records']:
        origin=item['duplicate_of'] or item['path'];r=final[origin]
        row={**r,**item,'boxes_xyxy_normalized':r['boxes_xyxy_normalized'],'status':r['status'],
             'annotation_origin':origin,'review_method':r['review_method'],'review_note':r['review_note']}
        if hashlib.sha256((raw/item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Raw file modified')
        all_rows.append(row)
    assert len(all_rows)==7343 and len(final)==7315
    selected=[r for r in all_rows if not r['duplicate_of'] and r['status']=='target_present']
    output.mkdir(parents=True,exist_ok=True)
    (output/'all_photo_annotations.json').write_text(json.dumps({'schema_version':2,'seed':42,'coordinate_system':'EXIF-oriented primary RGB frame 0; normalized xyxy','raw_sources_modified':False,'records':all_rows},indent=2)+'\n')
    (output/'positive_source_annotations.json').write_text(json.dumps({'schema_version':2,'seed':42,'coordinate_system':'EXIF-oriented primary RGB frame 0; normalized xyxy','selected_sources':selected},indent=2)+'\n')
    labels=output/'yolo_labels'
    for row in all_rows:
        path=labels/Path(row['path']).with_suffix('.txt');path.parent.mkdir(parents=True,exist_ok=True)
        lines=[]
        for x1,y1,x2,y2 in row['boxes_xyxy_normalized']:lines.append(f"{row['class_index']} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}\n")
        path.write_text(''.join(lines))
    summary={'raw_photos':7343,'unique_photos':7315,'duplicates_with_alias_annotations':28,'unique_status_counts':dict(counts),'positive_sources_for_synthesis':len(selected),'positive_source_split_counts':dict(Counter(r['split'] for r in selected)),'unique_positive_target_instances':sum(len(r['boxes_xyxy_normalized']) for r in selected),'explicit_review_overrides':len(overrides),'all_photo_label_files':len(list(labels.rglob('*.txt'))),'raw_sources_modified':False,'reviewed_sheets':len(checked),'annotation_scope':'Visible instances of the folder Object ID; incidental other classes not exhaustively labelled','independent_human_review':False}
    (output/'annotation_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','finalize']);p.add_argument('--raw',type=Path,required=True);p.add_argument('--proposals',type=Path);p.add_argument('--review',type=Path,required=True);p.add_argument('--decisions',type=Path);p.add_argument('--output',type=Path);p.add_argument('--inventory',type=Path,default=HERE/'annotations/all_source_inventory.json');a=p.parse_args()
    if a.action=='prepare':prepare(a.raw,a.proposals,a.review,a.inventory)
    else:finalize(a.raw,a.review,a.decisions,a.output,a.inventory)
