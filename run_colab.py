"""Upload a verified dataset to Colab CLI, train YOLOv8s, and recover exports."""
import argparse,hashlib,json,subprocess,sys,time,tarfile
from pathlib import Path
from datetime import datetime,timezone
import transfer
from export_helpers import package_archive
ROOT=Path(__file__).resolve().parent

def run_cli_checked(base,*args):
 # A disconnected CLI websocket can leave non-daemon threads alive after a traceback.
 # Stop that local client; never interrupt or shut down the remote training kernel here.
 process=subprocess.Popen([*base,*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 try:
  for line in process.stdout:
   print(transfer.sanitize_cli_output(line),end='',flush=True)
   if 'RuntimeError: Connection was lost.' in line:
    raise RuntimeError('Colab client websocket disconnected')
  if process.wait():raise RuntimeError(f'Colab {args[0]} failed')
 finally:
  if process.poll() is None:
   process.terminate()
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:process.kill();process.wait()
  process.stdout.close()

def retry_idempotent(action,attempts=5,delay=time.sleep):
 for attempt in range(attempts):
  try:return action()
  except RuntimeError:
   if attempt==attempts-1:raise
   print(f'Reconnecting for idempotent setup ({attempt+2}/{attempts})',flush=True)
   delay(min(2**attempt,8))

def upload_with_retries(cli,session,local,remote,attempts=5,delay=time.sleep):
 if attempts<1:raise ValueError('At least one upload attempt is required')
 for attempt in range(attempts):
  try:
   return cli('upload','-s',session,str(local),remote)
  except RuntimeError:
   if attempt==attempts-1:raise
   print(f'Retrying upload of {Path(local).name} ({attempt+2}/{attempts})',flush=True)
   delay(min(2**attempt,8))

def main():
 p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--gpu',default='A100',choices=['A100','L4','T4']);p.add_argument('--config',type=Path,default=ROOT/'.colab-sessions.json');p.add_argument('--dataset',type=Path,default=ROOT/'data/synthetic_seed42');p.add_argument('--output',type=Path,default=ROOT/'artifacts/yolov8s_seed42');p.add_argument('--attach',action='store_true');p.add_argument('--recover',action='store_true');a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);base=['colab','--config',str(a.config.resolve())]
 def cli(*args):return run_cli_checked(base,*args)
 def download(remote,destination,size=None,digest=None):
  temp=destination.with_name(destination.name+'.partial')
  for attempt in range(3):
   try:
    result=subprocess.run([*base,'download','-s',a.session,remote,str(temp)],capture_output=True,text=True,timeout=300)
    if result.returncode:raise RuntimeError('Colab download failed; authentication/proxy output suppressed')
    if size is not None and not transfer.valid_file(temp,size,digest):raise RuntimeError('Artifact checksum mismatch')
    temp.replace(destination);return
   except Exception:
    if attempt==2:raise
    time.sleep(2**attempt)
 started=a.attach or a.recover;protect=a.recover;verified=False
 try:
  if not a.recover:
   from validate_dataset import validate
   validate(a.dataset)
   if not a.attach:cli('new','-s',a.session,'--gpu',a.gpu);started=True
   (out/'session_receipt.json').write_text(json.dumps({'session':a.session,'gpu':a.gpu,'created_utc':datetime.now(timezone.utc).isoformat(),'exports_verified':False},indent=2)+'\n')
   files=[p for p in ROOT.glob('*.py')]+[ROOT/'requirements.txt',ROOT/'class_mapping.json']+list((ROOT/'tests').glob('*.py'))+list((ROOT/'annotations').glob('source_annotations.json'))
   entries=[(p,p.relative_to(ROOT).as_posix()) for p in files]+[(a.dataset,'data/synthetic_seed42')]
   package_archive(out/'input.tar.gz',entries)
   archive=out/'input.tar.gz';parts=out/'input.parts';parts.mkdir(exist_ok=True);manifest=[]
   with archive.open('rb') as f:
    while chunk:=f.read(32*1024*1024):
     index=len(manifest);name=f'part-{index:05d}';(parts/name).write_bytes(chunk);manifest.append({'name':name,'size':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
   manifestpath=out/'input_manifest.json';manifestpath.write_text(json.dumps({'parts':manifest,'archive_sha256':transfer.sha256(archive)},indent=2)+'\n')
   retry_idempotent(lambda:transfer.ensure_remote_directories(cli,a.session,out/'setup',['/content/milestone2/input.parts']))
   for part in manifest:upload_with_retries(cli,a.session,parts/part['name'],'/content/milestone2/input.parts/'+part['name'])
   upload_with_retries(cli,a.session,manifestpath,'/content/milestone2/input_manifest.json')
   receiver=out/'receive.py';receiver.write_text("import json,hashlib,tarfile\nfrom pathlib import Path\nr=Path('/content/milestone2');m=json.loads((r/'input_manifest.json').read_text());a=r/'input.tar.gz'\nwith a.open('wb') as f:\n for p in m['parts']:\n  b=(r/'input.parts'/p['name']).read_bytes();assert len(b)==p['size'] and hashlib.sha256(b).hexdigest()==p['sha256'];f.write(b)\nassert hashlib.sha256(a.read_bytes()).hexdigest()==m['archive_sha256']\nwith tarfile.open(a) as t:\n for item in t.getmembers():\n  dest=(r/item.name).resolve();assert r in dest.parents and (item.isfile() or item.isdir())\n t.extractall(r,filter='data')\na.unlink()\nprint('Dataset and source upload verified',flush=True)\n")
   retry_idempotent(lambda:cli('exec','-s',a.session,'-f',str(receiver),'--timeout','600'));protect=True
   cli('exec','-s',a.session,'-f',str(ROOT/'colab_worker.py'),'--timeout','21600')
  archive,manifest=transfer.retrieve_archive(download,out,'artifacts','experiment.tar.gz')
  destination=out/'experiment'
  if destination.exists():destination=out/('experiment_recovered_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
  transfer.unpack(archive,destination);complete=json.loads((destination/'runs/completed.json').read_text())
  if complete.get('completed') is not True or complete.get('epochs')!=50:raise RuntimeError('Artifacts do not show completed 50-epoch training')
  verified=True;print(f'Experiment completed and checksum-verified: {destination}',flush=True)
 finally:
  if started and (verified or not protect):
   cli('stop','-s',a.session)
   receipt=out/'session_receipt.json'
   if receipt.exists():
    value=json.loads(receipt.read_text());value.update(gpu_stopped=True,exports_verified=verified);receipt.write_text(json.dumps(value,indent=2)+'\n')
  elif started:print(f'Runtime retained for artifact recovery: {a.session}. Rerun with --recover and the same --config/output.',flush=True)
if __name__=='__main__':main()
