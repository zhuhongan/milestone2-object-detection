"""Detect Object IDs and pixel locations in a supplied multi-object image."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--source',required=True);p.add_argument('--output',type=Path,default=Path('predictions'));p.add_argument('--mapping',type=Path,default=Path(__file__).with_name('class_mapping.json'));p.add_argument('--threshold-file',type=Path);p.add_argument('--conf',type=float);p.add_argument('--device',default='cpu');a=p.parse_args()
 from ultralytics import YOLO
 mapping=json.loads(a.mapping.read_text());conf=a.conf if a.conf is not None else (json.loads(a.threshold_file.read_text())['confidence'] if a.threshold_file else .25)
 model=YOLO(str(a.weights));a.output.mkdir(parents=True,exist_ok=True);allresults=[]
 if [model.names[i] for i in range(73)]!=[m['object_id'] for m in mapping]:raise ValueError('Weights do not match final Object IDs')
 for r in model.predict(a.source,conf=conf,iou=.7,imgsz=640,device=a.device,stream=True,verbose=False):
  boxes=[]
  for i in range(len(r.boxes)):
   c=int(r.boxes.cls[i]);boxes.append({'object_id':mapping[c]['object_id'],'object_name':mapping[c]['object_name'],'confidence':float(r.boxes.conf[i]),'xyxy_pixels':[round(float(v),2) for v in r.boxes.xyxy[i].cpu().tolist()]})
  r.save(filename=str(a.output/Path(r.path).name));allresults.append({'image':r.path,'height':r.orig_shape[0],'width':r.orig_shape[1],'detections':boxes})
 (a.output/'detections.json').write_text(json.dumps(allresults,indent=2)+'\n');print(json.dumps(allresults,indent=2))
if __name__=='__main__':main()
