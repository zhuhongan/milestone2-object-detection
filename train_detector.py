"""Train the required YOLOv8 detector and export honest held-out evaluations."""
import argparse,csv,hashlib,json,os,platform,random,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def save_detections(results,mapping,output):
 rows=[]
 for result in results:
  for i in range(len(result.boxes)):
   c=int(result.boxes.cls[i]);box=[float(v) for v in result.boxes.xyxy[i].cpu().tolist()]
   rows.append({'image':Path(result.path).name,'class_index':c,'object_id':mapping[c]['object_id'],'object_name':mapping[c]['object_name'],'confidence':float(result.boxes.conf[i]),'x1':box[0],'y1':box[1],'x2':box[2],'y2':box[3]})
 fields=['image','class_index','object_id','object_name','confidence','x1','y1','x2','y2']
 with Path(output).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,default=HERE/'runs');p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch',type=int,default=32);p.add_argument('--device',default='0');p.add_argument('--resume',action='store_true');a=p.parse_args()
 import torch,ultralytics,yaml
 from ultralytics import YOLO
 dataset=a.dataset.resolve();output=a.output.resolve();output.mkdir(parents=True,exist_ok=True);mapping=json.loads((dataset/'class_mapping.json').read_text());audit=json.loads((dataset/'audit.json').read_text())
 if len(mapping)!=73 or audit['seed']!=42 or not audit['source_disjoint']:raise ValueError('Dataset verification failed')
 # Resolve YAML root explicitly: a portable relative data.yaml is not inferred from process cwd.
 cfg=yaml.safe_load((dataset/'data.yaml').read_text());cfg['path']=str(dataset);cfgpath=output/'resolved_data.yaml';cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False))
 write_json(output/'environment.json',{'python':platform.python_version(),'ultralytics':ultralytics.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'packages':subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines()})
 run=output/'yolov8s_seed42';model=YOLO(str(run/'weights/last.pt') if a.resume else 'yolov8s.pt')
 initial=Path('yolov8s.pt');pretrained_sha=digest(initial) if initial.exists() else None
 # One fixed budget and checkpoint selection based on validation mAP50-95.
 params=dict(data=str(cfgpath),epochs=a.epochs,imgsz=640,batch=a.batch,device=a.device,workers=4,seed=42,deterministic=True,optimizer='AdamW',lr0=.001,lrf=.01,cos_lr=True,weight_decay=.0005,warmup_epochs=3,patience=0,amp=True,cache=False,project=str(output),name='yolov8s_seed42',exist_ok=a.resume,save=True,save_period=10,plots=True,verbose=False,mosaic=1.0,close_mosaic=10,mixup=0.0,copy_paste=0.0,degrees=10.0,translate=.1,scale=.3,shear=0.0,perspective=0.0,fliplr=.5,flipud=0.0,hsv_h=.015,hsv_s=.3,hsv_v=.2)
 write_json(output/'training_protocol.json',{'model':'yolov8s.pt','pretrained_sha256':pretrained_sha,'selection':'Highest validation mAP50-95 (Ultralytics detection fitness); test excluded from selection','parameters':params,'source_annotations_sha256':audit['source_annotations_sha256'],'generated_manifest_sha256':digest(dataset/'manifest.jsonl')})
 def track_best(trainer):
  if trainer.fitness == trainer.best_fitness:
   write_json(output/'best_epoch_receipt.json',{'epoch':trainer.epoch+1,'fitness':float(trainer.fitness),'rule':'Library saves the latest tied best checkpoint'})
 model.add_callback('on_model_save',track_best)
 started=time.monotonic();model.train(resume=True) if a.resume else model.train(**params);elapsed=time.monotonic()-started
 history=list(csv.DictReader((run/'results.csv').open()));history=[{k.strip():v.strip() for k,v in r.items()} for r in history]
 if len(history)!=a.epochs:raise RuntimeError(f'Incomplete training: {len(history)} of {a.epochs} epochs')
 def score(r):return float(r['metrics/mAP50-95(B)'])
 selected_epoch=json.loads((output/'best_epoch_receipt.json').read_text())['epoch'];bestrow=next(r for r in history if int(float(r['epoch']))==selected_epoch);write_json(output/'selection.json',{'selected_epoch':int(float(bestrow['epoch'])),'validation_mAP50_95':score(bestrow),'rule':'Maximum validation mAP50-95; latest tied checkpoint, confirmed by save callback','checkpoint_sha256':digest(run/'weights/best.pt'),'test_accessed':False})
 best=YOLO(str(run/'weights/best.pt'));metrics={};perclass=[]
 for split in ['val','test']:
  result=best.val(data=str(cfgpath),split=split,imgsz=640,batch=a.batch,device=a.device,workers=4,conf=.001,iou=.7,max_det=100,plots=True,save_json=True,project=str(output),name=f'{split}_evaluation',exist_ok=False,verbose=False)
  metrics[split]={'precision':float(result.box.mp),'recall':float(result.box.mr),'mAP50':float(result.box.map50),'mAP50_95':float(result.box.map),'metric_confidence':'PR/AP curves evaluated from confidence 0.001; class-mean precision/recall at the best mean-F1 point on this evaluated split; separate operational threshold selected on validation only','nms_iou':.7,'images':audit['counts'][split]['images'],'target_instances':audit['counts'][split]['target_instances']}
  for j,c in enumerate(result.box.ap_class_index):
   c=int(c);perclass.append({'split':split,**{k:mapping[c][k] for k in ['class_index','object_id','object_name']},'precision':float(result.box.p[j]),'recall':float(result.box.r[j]),'AP50':float(result.box.ap50[j]),'AP50_95':float(result.box.ap[j])})
 write_json(output/'metrics.json',metrics)
 with (output/'per_class_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(perclass[0]));w.writeheader();w.writerows(perclass)
 # Threshold is chosen once from the validation prediction set, then fixed before test.
 valpred=best.predict(source=str(dataset/'images/val'),imgsz=640,conf=.01,iou=.7,max_det=100,device=a.device,stream=True,verbose=False)
 valrows=save_detections(valpred,mapping,output/'val_predictions_candidates.csv')
 from evaluate_predictions import evaluate,choose_threshold
 manifest=[json.loads(x) for x in (dataset/'manifest.jsonl').read_text().splitlines()]
 threshold,grid=choose_threshold(valrows,[r for r in manifest if r['split']=='val'])
 write_json(output/'operating_threshold.json',{'confidence':threshold,'selection':'Maximum validation micro-F1 at matching IoU 0.5; ties prefer higher threshold','validation_grid':grid,'nms_iou':.7,'match_iou':.5})
 testpred=best.predict(source=str(dataset/'images/test'),imgsz=640,conf=.01,iou=.7,max_det=100,device=a.device,stream=True,verbose=False)
 testrows=save_detections(testpred,mapping,output/'test_predictions_candidates.csv')
 fixedrows=[r for r in testrows if r['confidence']>=threshold]
 with (output/'test_detections.csv').open('w',newline='') as f:
  fields=['image','class_index','object_id','object_name','confidence','x1','y1','x2','y2'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(fixedrows)
 operating=evaluate(testrows,[r for r in manifest if r['split']=='test'],threshold)
 write_json(output/'test_operating_metrics.json',operating)
 # Report each layout separately using the same checkpoint and global val-selected threshold.
 if audit.get('layouts'):
  layout_metrics={};layout_perclass=[]
  for split,rows in [('val',valrows),('test',testrows)]:
   layout_metrics[split]={}
   for layout in audit['layouts']:
    group=[r for r in manifest if r['split']==split and r['layout']==layout]
    names={Path(r['image']).name for r in group}
    imagesfile=output/f'{split}_{layout}_images.txt'
    imagesfile.write_text(''.join(str(dataset/r['image'])+'\n' for r in group))
    layoutcfg={**cfg,split:str(imagesfile)}
    layoutcfgpath=output/f'{split}_{layout}_data.yaml'
    layoutcfgpath.write_text(yaml.safe_dump(layoutcfg,sort_keys=False))
    result=best.val(data=str(layoutcfgpath),split=split,imgsz=640,batch=a.batch,device=a.device,workers=4,conf=.001,iou=.7,max_det=100,plots=False,save_json=False,project=str(output),name=f'{split}_{layout}_evaluation',exist_ok=False,verbose=False)
    layout_metrics[split][layout]={'images':len(group),'target_instances':sum(len(r['targets']) for r in group),'precision':float(result.box.mp),'recall':float(result.box.mr),'mAP50':float(result.box.map50),'mAP50_95':float(result.box.map),'operating':evaluate([r for r in rows if r['image'] in names],group,threshold)}
    for j,c in enumerate(result.box.ap_class_index):
     c=int(c);layout_perclass.append({'split':split,'layout':layout,**{k:mapping[c][k] for k in ['class_index','object_id','object_name']},'precision':float(result.box.p[j]),'recall':float(result.box.r[j]),'AP50':float(result.box.ap50[j]),'AP50_95':float(result.box.ap[j])})
  write_json(output/'layout_metrics.json',layout_metrics)
  with (output/'layout_per_class_metrics.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(layout_perclass[0]));writer.writeheader();writer.writerows(layout_perclass)
 # Gallery paths selected independently of outcomes; plot all detections at the fixed threshold.
 images=sorted((dataset/'images/test').glob('*.jpg'));selected=random.Random(42).sample(images,6)
 if audit.get('layouts'):
  sampler=random.Random(42);selected=[]
  for layout in audit['layouts']:
   paths=sorted(dataset/r['image'] for r in manifest if r['split']=='test' and r['layout']==layout)
   selected.extend(sampler.sample(paths,2))
 gallery=output/'detection_examples';gallery.mkdir(exist_ok=True)
 write_json(gallery/'selection.json',{'seed':42,'rule':'Two sorted test paths sampled per layout, independent of predictions' if audit.get('layouts') else 'Uniform sample of six sorted test paths independent of predictions','files':[p.name for p in selected]})
 byimage={p.name:[] for p in images}
 for row in fixedrows:byimage[row['image']].append(row)
 try:font=ImageFont.truetype('DejaVuSans.ttf',16)
 except OSError:font=ImageFont.load_default()
 for imagepath in selected:
  im=Image.open(imagepath).convert('RGB');draw=ImageDraw.Draw(im)
  for row in byimage[imagepath.name]:
   b=[row[k] for k in ['x1','y1','x2','y2']];draw.rectangle(b,outline='#087e83',width=3)
   label=f"{row['object_id']} {row['confidence']:.2f}";origin=(max(0,b[0]),max(0,b[1]-20));bb=draw.textbbox(origin,label,font=font);draw.rectangle(bb,fill='white');draw.text(origin,label,font=font,fill='#005c61')
  im.save(gallery/imagepath.name,quality=95)
 write_json(output/'completed.json',{'completed':True,'model':'YOLOv8s','epochs':len(history),'training_seconds':elapsed,'selected_epoch':int(float(bestrow['epoch'])),'best_sha256':digest(run/'weights/best.pt'),'last_sha256':digest(run/'weights/last.pt'),'dataset_audit':audit,'metrics':metrics,'operating_metrics':operating,'confidence':threshold})
 print(json.dumps({'completed':True,'metrics':metrics,'operating':operating},indent=2),flush=True)
if __name__=='__main__':main()
