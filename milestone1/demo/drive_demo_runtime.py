"""Drive-folder inference runner embedded in both self-contained Colab demos."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import uuid


def write_state(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.temporary-'+uuid.uuid4().hex[:8])
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare_cached_model(source, destination, expected_hash):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination)==expected_hash:
        return destination
    if not source.is_file():
        raise FileNotFoundError(f'Model missing from configured Drive path: {source}')
    import shutil
    temporary = destination.with_suffix('.partial')
    shutil.copyfile(source, temporary)
    if sha256_file(temporary)!=expected_hash:
        raise ValueError('Checkpoint checksum differs from the selected experiment')
    temporary.replace(destination)
    return destination


def run_demo(process_all=False):
    """Find new/changed input photos, predict and persist results without moving inputs."""
    import pandas as pd
    from IPython.display import display, Image as DisplayImage
    supported = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.mpo'}
    INPUT_DIR.mkdir(parents=True,exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    state_path = OUTPUT_DIR/'processed_inputs.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'schema':1,'processed':{}}
    signature = EXPECTED_MODEL_SHA256 + (f':confidence={CONFIDENCE}' if MILESTONE==2 else ':eval256crop224')
    pending, old = [], 0
    candidates = sorted(p for p in INPUT_DIR.rglob('*') if p.is_file() and p.suffix.lower() in supported)
    for image in candidates:
        relative = image.relative_to(INPUT_DIR).as_posix()
        digest = sha256_file(image)
        previous = state['processed'].get(relative)
        if not process_all and previous and previous['sha256']==digest and previous['signature']==signature:
            old += 1
            continue
        pending.append((image,relative,digest))
    print(f'Input folder: {INPUT_DIR}')
    print(f'{len(pending)} new/changed image(s); {old} unchanged image(s) already processed.')
    if not pending:
        print('Upload the TA photos into the input folder, wait for the Drive upload to finish, then rerun this cell.')
        print('Previous results remain in:', OUTPUT_DIR)
        print('To repeat all practice images, run run_demo(process_all=True).')
        return {'processed':0,'skipped':old,'errors':[]}
    output = OUTPUT_DIR/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    records, errors = [], []
    for image,relative,digest in pending:
        try:
            if MILESTONE==1:
                record,rows,preview = classify_image(active_model,image,FINAL_MAPPING,output,
                    device=DEVICE,model_sha256=EXPECTED_MODEL_SHA256)
                top = record['prediction']
                print(f"{relative}: {top['object_id']} — {top['object_name']} (score {top['score']:.2%})")
            else:
                record,rows,preview = detect_image(active_model,image,FINAL_MAPPING,output,
                    confidence=CONFIDENCE,device=DEVICE,model_sha256=EXPECTED_MODEL_SHA256)
                print(f'{relative}: {len(rows)} predicted object(s), confidence ≥ {CONFIDENCE:.2f}')
            if sha256_file(image)!=digest:
                raise RuntimeError('Input changed during prediction; wait for the upload to finish and rerun')
            record['drive_input_relative_path'] = relative
            records.append(record)
            display(DisplayImage(filename=str(preview),width=min(900,record['width']) if MILESTONE==2 else 700))
            if rows:
                display(pd.DataFrame(rows).round({'confidence':3} if MILESTONE==2 else {'score':4}))
            else:
                print('No detections above the threshold. The model can miss objects.')
            state['processed'][relative] = {'sha256':digest,'signature':signature,
                'result_folder':output.name,'preview':preview.relative_to(OUTPUT_DIR).as_posix()}
            write_state(state_path,state)
        except Exception as error:
            errors.append({'input':relative,'error':str(error)})
            print(f'Could not process {relative}: {error}. It will be retried next time.')
    write_state(output/'batch_results.json',{'milestone':MILESTONE,'model_sha256':EXPECTED_MODEL_SHA256,
                 'records':records,'errors':errors})
    write_state(OUTPUT_DIR/'latest_run.json',{'result_folder':output.name,'processed':len(records),'errors':errors})
    print('Results saved permanently in Drive:',output)
    return {'processed':len(records),'skipped':old,'errors':errors,'output':str(output)}
