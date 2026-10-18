"""Run the benchmark and preserve verified exports even after a training failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,'/content/milestone1')
from colab_prepare_train import package_archive, split_archive
ROOT=Path('/content/milestone1')

def run(command,log_name):
    with (ROOT/log_name).open('a') as log:
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=ROOT)
        for line in child.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        if child.wait():raise RuntimeError(f'Command failed; see {log_name}')

def main():
    os.chdir(ROOT)
    try:
        run([sys.executable,'-m','pip','install','--quiet','timm==1.0.25'],'setup.log')
        run([sys.executable,'-m','unittest','test_prepare_dataset','test_download_dataset','benchmark.test_benchmark','-v'],'benchmark_checks.log')
        command=[sys.executable,'-u','benchmark/train_benchmark.py','--raw-root',str(ROOT/'data/final/images'),
                 '--prepared-dir',str(ROOT/'prepared'),'--output',str(ROOT/'benchmark_run')]
        if (ROOT/'benchmark_run/config.json').exists():command.append('--resume')
        run(command,'benchmark_training.log')
    finally:
        entries=[(p,p.name) for p in sorted(ROOT.iterdir()) if p.name not in ('data','__pycache__')]
        archive=Path('/content/milestone1_artifacts.tar.gz')
        package_archive(archive,entries)
        marker=ROOT/'benchmark_run/completed.json'
        split_archive(archive,'artifacts',json.loads(marker.read_text()) if marker.exists() else {'completed':False})

if __name__=='__main__':main()
