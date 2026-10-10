"""Verify completed artifacts and create report figures from saved detections."""
import argparse,csv,json,os,sys,shutil
from collections import defaultdict
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'artifacts/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from generate_dataset import sha256
from evaluate_predictions import evaluate,iou

HERE=Path(__file__).resolve().parent

def readcsv(path):return [{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(Path(path).open())]
def font(size):
 try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',size)
 except OSError:
  try:return ImageFont.truetype('DejaVuSans.ttf',size)
  except OSError:return ImageFont.load_default()
def draw_boxes(image,items,color,confidence=False):
 out=image.copy();draw=ImageDraw.Draw(out);labels=[]
 for item in items:
  b=item['xyxy'] if 'xyxy' in item else [float(item[k]) for k in ['x1','y1','x2','y2']]
  draw.rectangle(b,outline=color,width=4);label=item['object_id']+(f" {float(item['confidence']):.2f}" if confidence else '')
  bounds=draw.textbbox((0,0),label,font=font(27));w,h=bounds[2]-bounds[0],bounds[3]-bounds[1]
  candidates=[(b[0],b[1]-33),(b[0],b[1]-66),(b[0],b[3]+5),(b[2]-w,b[1]-33),(b[0],b[1]+5)]
  for cx,cy in candidates:
   x=min(max(2,cx),out.width-w-2);y=min(max(2,cy),out.height-h-bounds[1]-2)
   box=draw.textbbox((x,y),label,font=font(27));box=(box[0]-2,box[1]-2,box[2]+2,box[3]+2)
   if not any(box[0]<r[2]+3 and box[2]>r[0]-3 and box[1]<r[3]+3 and box[3]>r[1]-3 for r in labels):break
  labels.append(box);draw.rectangle(box,fill='white');draw.text((x,y),label,font=font(27),fill=color)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--dataset',type=Path,default=HERE/'data/mixed_layouts_seed42');p.add_argument('--output',type=Path,default=HERE/'reports/mixed_layouts_seed42');p.add_argument('--annotations',type=Path,default=HERE/'annotations/all_photos_seed42/positive_source_annotations.json');p.add_argument('--annotation-summary',type=Path,default=HERE/'annotations/all_photos_seed42/annotation_summary.json');a=p.parse_args()
 run=a.experiment.resolve()/'runs';dataset=a.dataset.resolve();out=a.output.resolve();plots=out/'plots';plots.mkdir(parents=True,exist_ok=True)
 complete=json.loads((run/'completed.json').read_text());metrics=json.loads((run/'metrics.json').read_text());selection=json.loads((run/'selection.json').read_text());operating=json.loads((run/'test_operating_metrics.json').read_text());env=json.loads((run/'environment.json').read_text());audit=json.loads((dataset/'audit.json').read_text());mapping=json.loads((dataset/'class_mapping.json').read_text());receipt=json.loads((run/'best_epoch_receipt.json').read_text());history=readcsv(run/'yolov8s_seed42/results.csv');perclass=readcsv(run/'per_class_metrics.csv');preds=readcsv(run/'test_predictions_candidates.csv');fixed=[r for r in preds if float(r['confidence'])>=operating['confidence']];manifest=[json.loads(s) for s in (dataset/'manifest.jsonl').read_text().splitlines()]
 assert complete['completed'] and len(history)==complete['epochs']==50
 assert sha256(run/'yolov8s_seed42/weights/best.pt')==complete['best_sha256']==selection['checkpoint_sha256']
 assert receipt['epoch']==selection['selected_epoch']==complete['selected_epoch']
 assert abs(max(float(r['metrics/mAP50-95(B)']) for r in history)-selection['validation_mAP50_95'])<1e-5
 assert complete['dataset_audit']==audit
 protocol=json.loads((run/'training_protocol.json').read_text())
 assert protocol['generated_manifest_sha256']==sha256(dataset/'manifest.jsonl')
 assert protocol['source_annotations_sha256']==sha256(a.annotations)
 recomputed=evaluate(preds,[r for r in manifest if r['split']=='test'],operating['confidence']);assert recomputed==operating
 for split in ['val','test']:
  rows=[r for r in perclass if r['split']==split];assert len(rows)==73;assert {r['object_id'] for r in rows}=={m['object_id'] for m in mapping}
  assert abs(np.mean([float(r['AP50_95']) for r in rows])-metrics[split]['mAP50_95'])<1e-10
  assert abs(np.mean([float(r['AP50']) for r in rows])-metrics[split]['mAP50'])<1e-10
 byimage=defaultdict(list)
 for r in fixed:byimage[r['image']].append(r)
 truth={r['image'].split('/')[-1]:r for r in manifest if r['split']=='test'}
 selected=json.loads((run/'detection_examples/selection.json').read_text())['files'];assert len(selected)==6
 # Four examples retain the ordering of the preselected six-image sample.
 gallery=Image.new('RGB',(1280,1390),'white');draw=ImageDraw.Draw(gallery)
 gallery_files=[selected[j] for j in [0,2,4,1]] if audit.get('layouts') else selected[:4]
 for j,name in enumerate(gallery_files):
  image=Image.open(dataset/'images/test'/name).convert('RGB');image=draw_boxes(image,byimage[name],'#087e83',True);x=j%2*640;y=j//2*695
  im_metrics=next(r for r in operating['per_image'] if r['image']==name)
  label=f"{name}  {truth[name].get('layout','')}  TP {im_metrics['tp']} FP {im_metrics['fp']} FN {im_metrics['fn']}"
  draw.text((x+8,y+4),label,font=font(21),fill='black');gallery.paste(image,(x,y+48))
 gallery.save(plots/'test_examples.png')
 generation=Image.new('RGB',(1280,670),'white');draw=ImageDraw.Draw(generation)
 generation_rows=[next(r for r in manifest if r['split']=='train' and r['layout']==layout) for layout in ['2x2','3x3']] if audit.get('layouts') else manifest[:2]
 for j,row in enumerate(generation_rows):
  im=Image.open(dataset/row['image']).convert('RGB');generation.paste(draw_boxes(im,row['targets'],'#b62323'),(j*640,30));draw.text((j*640+8,4),Path(row['image']).name,font=font(23),fill='black')
 generation.save(plots/'generation_examples.png')
 # Failure galleries are explicitly chosen for errors; show ground truth beside prediction.
 errors=sorted([r for r in operating['per_image'] if r['fn'] or r['fp']],key=lambda r:(-r['fn'],-r['fp'],r['image']))[:2]
 failure_confusions=[]
 if errors:
  targets=truth[errors[0]['image']]['targets']
  for pred in byimage[errors[0]['image']]:
   box=[float(pred[k]) for k in ['x1','y1','x2','y2']]
   overlaps=[(iou(box,target['xyxy']),target) for target in targets]
   if not overlaps:continue
   overlap,target=max(overlaps,key=lambda pair:pair[0])
   if overlap>=.5 and pred['object_id']!=target['object_id'] and not any(score>=.5 and gt['object_id']==pred['object_id'] for score,gt in overlaps):
    failure_confusions.append({'target_object_id':target['object_id'],'predicted_object_id':pred['object_id']})
 failure=Image.new('RGB',(1280,1390),'white');draw=ImageDraw.Draw(failure)
 for j,row in enumerate(errors):
  name=row['image'];im=Image.open(dataset/'images/test'/name).convert('RGB');y=j*695
  draw.text((8,y+4),f"{name}  Ground truth",font=font(23),fill='black');draw.text((648,y+4),f"Predictions   TP {row['tp']} FP {row['fp']} FN {row['fn']}",font=font(23),fill='black')
  failure.paste(draw_boxes(im,truth[name]['targets'],'#b62323'),(0,y+48));failure.paste(draw_boxes(im,byimage[name],'#087e83',True),(640,y+48))
 failure.save(plots/'failure_examples.png')
 failure.crop((0,0,1280,695)).save(plots/'failure_example_one.png')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 epochs=np.array([int(float(r['epoch'])) for r in history]);fig,axes=plt.subplots(2,2,figsize=(9,5),layout='constrained')
 for ax,key,label in [(axes[0,0],'box_loss','Box loss'),(axes[0,1],'cls_loss','Classification loss')]:
  for prefix,color in [('train/','#2876b5'),('val/','#d77b2e')]:ax.plot(epochs,[float(r[prefix+key]) for r in history],label=prefix[:-1],color=color)
  ax.set_title(label);ax.legend(frameon=False);ax.set_xlabel('Epoch')
 for ax,key,label in [(axes[1,0],'metrics/mAP50(B)','Validation mAP50'),(axes[1,1],'metrics/mAP50-95(B)','Validation mAP50–95')]:
  ax.plot(epochs,[float(r[key]) for r in history],color='#087e83');ax.axvline(selection['selected_epoch'],color='#516474',ls='--',label=f"Selected epoch {selection['selected_epoch']}");ax.set_title(label);ax.set_xlabel('Epoch');ax.legend(frameon=False);ax.set_ylim(0,1)
 for ax in axes.flat:ax.grid(alpha=.15)
 fig.savefig(plots/'learning_curves.png',dpi=200);plt.close(fig)
 worst=sorted([r for r in perclass if r['split']=='test'],key=lambda r:(float(r['AP50_95']),r['object_id']))[:10]
 fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained');vals=[float(r['AP50_95'])*100 for r in worst]
 ax.barh([r['object_id']+'  '+r['object_name'] for r in worst],vals,color='#087e83');ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Test AP50–95 (%)');ax.set_title('Ten lowest class AP scores')
 for i,v in enumerate(vals):ax.text(v+1,i,f'{v:.1f}%',va='center',fontsize=9)
 fig.savefig(plots/'per_class_ap.png',dpi=200);plt.close(fig)
 annotation_records=json.loads(a.annotations.read_text())['selected_sources']
 data={'repository':'https://github.com/zhuhongan/milestone2-object-detection','complete':complete,'metrics':metrics,'operating':operating,'selection':selection,'environment':env,'audit':audit,'worst_classes':worst,'sample_files':selected,'sample_coordinates':byimage[selected[0]],'sample_image_metrics':next(r for r in operating['per_image'] if r['image']==selected[0]),'failure_examples':errors,'failure_class_confusions':failure_confusions,'class_mapping':mapping,'verified':True,'generated_dataset_manifest_sha256':sha256(dataset/'manifest.jsonl'),'best_checkpoint_sha256':complete['best_sha256'],'source_annotations_relative_path':a.annotations.resolve().relative_to(HERE).as_posix(),'annotation_summary':json.loads(a.annotation_summary.read_text()) if a.annotation_summary else {'positive_sources_for_synthesis':len(annotation_records),'unique_positive_target_instances':sum(len(r['boxes_xyxy_normalized']) for r in annotation_records)},'per_class_source_counts':{split:{m['object_id']:sum(r['split']==split and r['object_id']==m['object_id'] for r in annotation_records) for m in mapping} for split in ['train','val','test']}}
 data['gallery_files']=gallery_files;data['gallery_selection_rule']=json.loads((run/'detection_examples/selection.json').read_text())['rule']
 if audit.get('layouts'):
  layouts=json.loads((run/'layout_metrics.json').read_text());layoutclasses=readcsv(run/'layout_per_class_metrics.csv')
  for split in ['val','test']:
   allpreds=readcsv(run/f'{split}_predictions_candidates.csv')
   for layout in audit['layouts']:
    group=[r for r in manifest if r['split']==split and r['layout']==layout];names={Path(r['image']).name for r in group}
    assert evaluate([r for r in allpreds if r['image'] in names],group,operating['confidence'])==layouts[split][layout]['operating']
    rows=[r for r in layoutclasses if r['split']==split and r['layout']==layout];assert len(rows)==73
    for field,key in [('AP50','mAP50'),('AP50_95','mAP50_95')]:assert abs(np.mean([float(r[field]) for r in rows])-layouts[split][layout][key])<1e-10
  data['layout_metrics']=layouts
  fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained');x=np.arange(3);width=.25
  for offset,key,label,color in [(-width,'mAP50','mAP50','#9aadb8'),(0,'mAP50_95','mAP50–95','#087e83'),(width,'all_objects_accuracy','All objects correct','#d77b2e')]:
   values=[layouts['test'][layout]['operating'][key] if key=='all_objects_accuracy' else layouts['test'][layout][key] for layout in audit['layouts']]
   bars=ax.bar(x+offset,np.array(values)*100,width,label=label,color=color);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
  ax.set_xticks(x,['2×2','3×3','Random']);ax.set_ylim(0,120);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Score (%)');ax.set_title('Held-out performance by layout');ax.legend(frameon=False,loc='upper center',ncol=3,fontsize=9);ax.grid(axis='y',alpha=.15);fig.savefig(plots/'layout_performance.png',dpi=200);plt.close(fig)
  for name in ['layout_metrics.json','layout_per_class_metrics.csv']:shutil.copyfile(run/name,out/name)
  samples=HERE/'samples/mixed_layouts_seed42';samples.mkdir(parents=True,exist_ok=True);sample_index={}
  for j,layout in enumerate(['2x2','3x3','random']):
   name=selected[j*2];row=truth[name]
   assert row['layout']==layout
   shutil.copyfile(dataset/row['image'],samples/f'{layout}.jpg')
   draw_boxes(Image.open(dataset/row['image']).convert('RGB'),row['targets'],'#b62323').save(samples/f'{layout}_ground_truth.png')
   draw_boxes(Image.open(dataset/row['image']).convert('RGB'),byimage[name],'#087e83',True).save(samples/f'{layout}_detections.png')
   sample_index[layout]={'source_image':row['image'],'selection':'First of two seed-42 test paths sampled per layout independently of predictions','targets':row['targets'],'operating':next(r for r in operating['per_image'] if r['image']==name),'detections':byimage[name]}
  (samples/'samples.json').write_text(json.dumps(sample_index,indent=2)+'\n')
 (out/'report_data.json').write_text(json.dumps(data,indent=2)+'\n')
 for name in ['metrics.json','selection.json','operating_threshold.json','test_operating_metrics.json','per_class_metrics.csv','environment.json','training_protocol.json']:
  (out/name).write_bytes((run/name).read_bytes())
 evidence=out/'evidence';evidence.mkdir(exist_ok=True)
 for source,name in [(dataset/'manifest.jsonl','generated_manifest.jsonl'),(dataset/'audit.json','dataset_audit.json'),(dataset/'validation.json','dataset_validation.json'),(run/'yolov8s_seed42/results.csv','training_history.csv'),(run/'best_epoch_receipt.json','best_epoch_receipt.json'),(run/'val_predictions_candidates.csv','val_predictions_candidates.csv'),(run/'test_predictions_candidates.csv','test_predictions_candidates.csv'),(run/'test_detections.csv','test_detections.csv'),(a.experiment.parent/'session_receipt.json','session_receipt.json')]:
  shutil.copyfile(source,evidence/name)
 (evidence/'training_source_sha256.json').write_text(json.dumps({'annotations':a.annotations.resolve().relative_to(HERE).as_posix(),'sha256':sha256(a.annotations)},indent=2)+'\n')
 (out/'verification.json').write_text(json.dumps({'completed_50_epochs':True,'checkpoint_checksums':True,'selection_peak_confirmed':True,'official_class_mapping':True,'73_test_class_AP_scores':True,'independent_prediction_metric_recomputation':True},indent=2)+'\n')
 print(json.dumps({'report_assets_ready':str(out),'test':metrics['test'],'operating':{k:v for k,v in operating.items() if k!='per_image'}},indent=2))
if __name__=='__main__':main()
