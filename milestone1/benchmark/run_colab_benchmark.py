"""Colab CLI launch/recovery for the common four-model experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_colab as transfer
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--session',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='A100',choices=['A100','L4','T4'])
    p.add_argument('--recover',action='store_true')
    p.add_argument('--prepared-dir',type=Path,default=ROOT/'prepared')
    p.add_argument('--raw-snapshot',type=Path,default=ROOT/'data/raw/final-local.tar.gz')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=args.recover)
    if not args.recover:
        if not args.raw_snapshot.is_file():raise FileNotFoundError(f'Raw snapshot missing: {args.raw_snapshot}; acquire the finalized dataset first')
        for name in ('audit.json','class_mapping.json','train.csv','val.csv','test.csv'):
            if not (args.prepared_dir/name).is_file():raise FileNotFoundError(f'Prepared manifest missing: {args.prepared_dir/name}')
    base=['colab','--config',str(ROOT/'.colab-benchmark-sessions.json')]
    def cli(*a):return transfer.run_cli(base,*a,cwd=ROOT)
    started=args.recover;preserve=args.recover;verified=False
    def download(remote,destination,size=None,digest=None):
        temporary=destination.with_name(destination.name+'.partial')
        for attempt in range(3):
            try:
                result=subprocess.run([*base,'download','-s',args.session,remote,str(temporary)],capture_output=True,text=True,timeout=300)
                if result.returncode:raise RuntimeError('Colab transfer failed (proxy details suppressed)')
                if size is not None and not transfer.valid_file(temporary,size,digest):raise RuntimeError('Checksum mismatch')
                temporary.replace(destination);return
            except Exception:
                if attempt==2:raise
                time.sleep(2**attempt)
    try:
        if not args.recover:
            # Unique task-owned name; never overwrite another session registration.
            config=ROOT/'.colab-benchmark-sessions.json'
            if config.exists() and args.session in json.loads(config.read_text()):raise ValueError('Session name already registered')
            cli('new','-s',args.session,'--gpu',args.gpu);started=True
            (out/'session_receipt.json').write_text(json.dumps({'session':args.session,'gpu':args.gpu,'created_utc':datetime.now(timezone.utc).isoformat()},indent=2))
            transfer.ensure_remote_directories(cli,args.session,out/'setup',['/content/milestone1/benchmark','/content/milestone1/prepared'])
            files=['train.py','test_training.py','prepare_dataset.py','test_prepare_dataset.py','download_dataset.py','test_download_dataset.py',
                   'colab_prepare_train.py','benchmark/train_benchmark.py','benchmark/test_benchmark.py','benchmark/colab_worker.py','benchmark/source_inventory.json',
                   'benchmark/run_colab_benchmark.py','benchmark/analyze_benchmark.py','benchmark/render_report.py','benchmark/README.md','run_colab.py','colab_receive_raw.py']
            for name in files:cli('upload','-s',args.session,str(ROOT/name),'/content/milestone1/'+name)
            prepared=args.prepared_dir.resolve()
            for path in sorted(prepared.iterdir()):
                if path.is_file():cli('upload','-s',args.session,str(path),'/content/milestone1/prepared/'+path.name)
            transfer.upload_snapshot(cli,args.session,args.raw_snapshot.resolve(),out/'raw_upload')
            preserve=True
            cli('exec','-s',args.session,'-f',str(ROOT/'benchmark/colab_worker.py'),'--timeout','21600')
        archive,manifest=transfer.retrieve_archive(download,out,'artifacts','benchmark.tar.gz')
        verified=True
        transfer.unpack(archive,out/'experiment')
        complete=json.loads((out/'experiment/benchmark_run/completed.json').read_text())
        if not complete.get('pretrained') or complete.get('epochs_per_model')!=20:raise RuntimeError('Saved artifacts do not show a completed real20epoch run')
        print(f'Benchmark completed and verified at {out}',flush=True)
    finally:
        if started and (verified or not preserve):
            cli('stop','-s',args.session)
            receipt=out/'session_receipt.json'
            if receipt.exists():
                value=json.loads(receipt.read_text());value['gpu_stopped']=True;value['exports_verified']=verified
                receipt.write_text(json.dumps(value,indent=2)+'\n')
        elif started:
            print(f'Export recovery required. Runtime retained: {args.session}. Run this launcher with --recover and the same session/output.',flush=True)

if __name__=='__main__':main()
