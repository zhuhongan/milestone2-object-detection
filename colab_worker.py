"""Run the checked detector workload and preserve exports even on failure."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/content/milestone2')
def run(args,logname):
 with (ROOT/logname).open('a') as log:
  child=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=ROOT)
  for line in child.stdout:
   log.write(line);log.flush()
   # Progress lines are saved in full; forward concise epoch summaries and errors.
   if '\r' not in line and ('Traceback' in line or 'Error' in line or 'epoch' in line.lower() or 'completed' in line or 'all ' in line or 'Export' in line):print(line,end='',flush=True)
  if child.wait():raise RuntimeError(f'Command failed; see {logname}')
def main():
 os.chdir(ROOT)
 from export_helpers import package_archive,split_archive
 try:
  run([sys.executable,'-m','pip','install','--quiet','-r','requirements.txt'],'setup.log')
  run([sys.executable,'-m','unittest','discover','-s','tests','-v'],'tests.log')
  run([sys.executable,'validate_dataset.py','data/synthetic_seed42'],'validation.log')
  args=[sys.executable,'-u','train_detector.py','--dataset','data/synthetic_seed42','--output','runs','--epochs','50','--batch','32']
  if (ROOT/'runs/yolov8s_seed42/weights/last.pt').exists():args.append('--resume')
  run(args,'training.log')
 finally:
  entries=[(p,p.name) for p in sorted(ROOT.iterdir()) if p.name not in {'data','input.parts','__pycache__'}]
  marker=ROOT/'runs/completed.json';summary=json.loads(marker.read_text()) if marker.exists() else {'completed':False}
  archive=Path('/content/milestone2_artifacts.tar.gz');package_archive(archive,entries);split_archive(archive,'artifacts',summary)
if __name__=='__main__':main()
