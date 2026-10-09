"""Run an isolated resumable all-photo proposal job, retrieve verified outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import sys
import time
import transfer

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--config', type=Path, default=HERE / '.colab-annotation-sessions.json')
    p.add_argument('--output', type=Path, default=HERE / 'artifacts/all_photo_annotations')
    p.add_argument('--attach', action='store_true')
    p.add_argument('--skip-upload', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--keep', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cli_path = shutil.which('colab')
    if not cli_path:
        raise RuntimeError('Install authenticated Colab CLI first')
    base = [cli_path, '--config', str(a.config.resolve())]
    def cli(*args):
        transfer.run_cli(base, *args, cwd=HERE)
    def download(remote, path):
        for attempt in range(3):
            result = subprocess.run([*base, 'download', '-s', a.session, remote, str(path)],
                                    capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return
            if attempt == 2:
                raise RuntimeError('Colab download failed; credential/proxy output suppressed')
            time.sleep(attempt + 1)
    verified = False
    allocated = a.attach
    try:
        if not a.attach:
            cli('new', '-s', a.session, '--gpu', 'A100')
            allocated = True
        receipt = out / 'session_receipt.json'
        receipt.write_text(json.dumps({'session': a.session, 'gpu': 'A100',
                                      'created_utc': datetime.now(timezone.utc).isoformat(),
                                      'exports_verified': False, 'gpu_stopped': False}, indent=2) + '\n')
        if not a.skip_upload:
            inventory = json.loads((HERE / 'annotations/all_source_inventory.json').read_text())
            archive = out / 'input.tar.gz'
            with tarfile.open(archive, 'w:gz') as t:
                for row in inventory['records']:
                    if not row['duplicate_of']:
                        t.add(a.raw / row['path'], arcname='images/' + row['path'], recursive=False)
                for path, arcname in [(HERE / 'annotations/all_source_inventory.json', 'all_source_inventory.json'),
                                      (HERE / 'annotation_prompts.json', 'annotation_prompts.json'),
                                      (HERE / 'propose_all_annotations.py', 'propose_all_annotations.py')]:
                    t.add(path, arcname=arcname)
            parts = []
            with archive.open('rb') as f:
                while block := f.read(32 * 1024 * 1024):
                    path = out / f'input-part-{len(parts):03d}'
                    path.write_bytes(block)
                    parts.append({'name': path.name, 'sha256': hashlib.sha256(block).hexdigest(), 'size': len(block)})
            manifest = out / 'input_manifest.json'
            manifest.write_text(json.dumps({'parts': parts, 'sha256': transfer.sha256(archive)}, indent=2) + '\n')
            transfer.ensure_remote_directories(cli, a.session, out / 'setup', ['/content/milestone2/annotation_job/input'])
            for part in parts:
                cli('upload', '-s', a.session, str(out / part['name']), '/content/milestone2/annotation_job/input/' + part['name'])
            cli('upload', '-s', a.session, str(manifest), '/content/milestone2/annotation_job/input_manifest.json')
        worker = out / 'worker.py'
        worker.write_text("""import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
root=Path('/content/milestone2/annotation_job')
if (root/'input_manifest.json').exists() and not (root/'propose_all_annotations.py').exists():
 m=json.loads((root/'input_manifest.json').read_text());archive=root/'received.tar.gz'
 with archive.open('wb') as f:
  for p in m['parts']:
   b=(root/'input'/p['name']).read_bytes();assert len(b)==p['size'] and hashlib.sha256(b).hexdigest()==p['sha256'];f.write(b)
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['sha256']
 with tarfile.open(archive) as t:
  for member in t.getmembers():assert root in (root/member.name).resolve().parents and (member.isfile() or member.isdir())
  t.extractall(root,filter='data')
 print('All source upload checksums verified',flush=True)
subprocess.run([sys.executable,'-m','pip','install','--quiet','transformers==4.57.1','huggingface-hub==0.36.0','Pillow==11.3.0','numpy==2.1.3'],check=True)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
args=[sys.executable,str(root/'propose_all_annotations.py'),'--root',str(root)]
""" + (f"args.extend(['--limit',{str(a.limit)!r}])\n" if a.limit else '') + """try:
 subprocess.run(args,check=True)
finally:
 output=root/'output';archive=root/'annotation_results.tar.gz'
 if output.exists():
  with tarfile.open(archive,'w:gz') as t:t.add(output,arcname='output')
  receipt={'size':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
  (root/'annotation_export.json').write_text(json.dumps(receipt))
  print('Annotation export ready',receipt,flush=True)
""")
        cli('exec', '-s', a.session, '-f', str(worker), '--timeout', '21600')
        download('/content/milestone2/annotation_job/annotation_export.json', out / 'annotation_export.json')
        export = json.loads((out / 'annotation_export.json').read_text())
        archive = out / ('smoke_results.tar.gz' if a.limit else 'annotation_results.tar.gz')
        download('/content/milestone2/annotation_job/annotation_results.tar.gz', archive)
        assert transfer.valid_file(archive, export['size'], export['sha256'])
        dest = out / ('smoke' if a.limit else 'full')
        transfer.unpack(archive, dest)
        marker = dest / 'output' / ('smoke_completed.json' if a.limit else 'completed.json')
        assert json.loads(marker.read_text())['completed']
        verified = True
        value = json.loads(receipt.read_text())
        value['exports_verified'] = True
        receipt.write_text(json.dumps(value, indent=2) + '\n')
        print('Verified annotation proposals:', dest, flush=True)
    finally:
        if allocated and verified and not a.keep:
            cli('stop', '-s', a.session)
            value = json.loads(receipt.read_text())
            value['gpu_stopped'] = True
            receipt.write_text(json.dumps(value, indent=2) + '\n')
        elif allocated:
            print('Owned GPU retained for the next phase or recovery:', a.session, flush=True)


if __name__ == '__main__':
    main()
