"""Four ImageNet backbones, identical manifests/head/augmentation and 5+15 epochs."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as common
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

MODELS = ('vgg16', 'resnet50', 'mobilenetv2_140', 'convnext_tiny')
WEIGHTS = {'vgg16':'torchvision.VGG16_Weights.IMAGENET1K_V1',
           'resnet50':'torchvision.ResNet50_Weights.IMAGENET1K_V1',
           'mobilenetv2_140':'timm.mobilenetv2_140.ra_in1k',
           'convnext_tiny':'torchvision.ConvNeXt_Tiny_Weights.IMAGENET1K_V1'}

class Classifier(nn.Module):
    def __init__(self, name, classes, pretrained=True):
        super().__init__()
        if name == 'vgg16':
            source = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone = nn.Sequential(source.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            dim = 512
        elif name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
        elif name == 'mobilenetv2_140':
            import timm
            self.backbone = timm.create_model('mobilenetv2_140.ra_in1k', pretrained=pretrained, num_classes=0)
            dim = self.backbone.num_features
            cfg = self.backbone.pretrained_cfg
            if tuple(cfg['mean']) != common.MEAN or tuple(cfg['std']) != common.STD:
                raise ValueError('Pretrained normalization differs from the declared shared normalization')
        else:
            raise ValueError(name)
        self.head = nn.Sequential(nn.Linear(dim,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.3), nn.Linear(256,classes))
        self.frozen = False

    def set_phase(self, frozen):
        self.frozen = frozen
        for p in self.backbone.parameters(): p.requires_grad_(not frozen)
        for p in self.head.parameters(): p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen: self.backbone.eval()
        return self

    def forward(self, x):
        return self.head(self.backbone(x))

def loaders_for(args, manifests, epoch=0):
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.7,1), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(.5), transforms.ColorJitter(.1,.1,.1,.02),
        transforms.ToTensor(), transforms.Normalize(common.MEAN,common.STD)])
    evaluation = transforms.Compose([
        transforms.Resize(256,interpolation=InterpolationMode.BILINEAR,antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(common.MEAN,common.STD)])
    return {split: DataLoader(common.ManifestDataset(args.raw_root, rows, augmentation if split=='train' else evaluation),
             batch_size=args.batch_size, shuffle=split=='train', num_workers=args.workers,
             pin_memory=args.device=='cuda', worker_init_fn=common.seed_worker, drop_last=False,
             generator=torch.Generator().manual_seed(args.seed + 1000*epoch + index))
            for index,(split,rows) in enumerate(manifests.items())}

def optimizer_for(model, phase, epoch_in_phase, phase_epochs):
    factor = .5*(1+math.cos(math.pi*epoch_in_phase/phase_epochs))
    if phase=='frozen':
        return torch.optim.AdamW([{'params':model.head.parameters(),'lr':1e-3*factor}],weight_decay=.01)
    return torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5*factor},
                             {'params':model.head.parameters(),'lr':1e-4*factor}],weight_decay=.01)

def phase_at(epoch, frozen_epochs):
    return 'frozen' if epoch<=frozen_epochs else 'finetuned'

def fit_one(name, args, manifests, classes, config_hash):
    directory=args.output/name
    directory.mkdir(exist_ok=True)
    last_path=directory/'last.pt'
    common.seed_everything(args.seed)
    model=Classifier(name,len(classes),pretrained=not args.smoke and not last_path.exists()).to(args.device)
    history=[]; best=None; start=1; previous_phase=None; optimizer=None
    scaler=torch.amp.GradScaler('cuda',enabled=common.precision_for(args.device)==torch.float16,init_scale=1024)
    if last_path.exists():
        state=common.load_checkpoint(last_path)
        if state['config_hash']!=config_hash: raise ValueError('Checkpoint configuration changed')
        model.load_state_dict(state['model'])
        history,best,start=state['history'],state['best'],state['epoch']+1
        previous_phase=state['phase']
        model.set_phase(previous_phase=='frozen')
        phase_epochs=args.frozen_epochs if previous_phase=='frozen' else args.epochs-args.frozen_epochs
        optimizer=optimizer_for(model,previous_phase,0,phase_epochs)
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
    for epoch in range(start,args.epochs+1):
        common.seed_everything(args.seed+epoch)
        loaders=loaders_for(args,manifests,epoch)
        phase=phase_at(epoch,args.frozen_epochs)
        phase_epoch=epoch-1 if phase=='frozen' else epoch-args.frozen_epochs-1
        phase_epochs=args.frozen_epochs if phase=='frozen' else args.epochs-args.frozen_epochs
        model.set_phase(phase=='frozen')
        if phase!=previous_phase:
            optimizer=optimizer_for(model,phase,phase_epoch,phase_epochs)
        factor=.5*(1+math.cos(math.pi*phase_epoch/phase_epochs))
        for g,base_lr in zip(optimizer.param_groups,[1e-3] if phase=='frozen' else [1e-5,1e-4]): g['lr']=base_lr*factor
        print(f'{name}: epoch {epoch}/{args.epochs}, {phase}, LR={[g["lr"] for g in optimizer.param_groups]}',flush=True)
        torch.cuda.synchronize() if args.device=='cuda' else None
        began=time.monotonic()
        # The wrapper owns frozen-backbone mode; shared epoch loop treats the head as trainable.
        tr,_,_=common.run_epoch(model,loaders['train'],torch.device(args.device),len(classes),'benchmark',100,optimizer,scaler)
        va,_,_=common.run_epoch(model,loaders['val'],torch.device(args.device),len(classes),'benchmark',100)
        if common.better(va,best):
            best=dict(va,epoch=epoch,phase=phase)
            common.atomic_torch_save(directory/'best.pt',{'model':model.state_dict(),'architecture':name,
                'classes':classes,'epoch':epoch,'validation':va,'config_hash':config_hash,'pretrained':not args.smoke})
        torch.cuda.synchronize() if args.device=='cuda' else None
        row={'epoch':epoch,'phase':phase,'seconds':time.monotonic()-began,
             **{f'train_{k}':tr[k] for k in ('loss','accuracy','macro_f1')},
             **{f'val_{k}':va[k] for k in ('loss','accuracy','macro_f1')},
             'optimizer_updates':tr['optimizer_updates'],'skipped_optimizer_steps':tr['skipped_optimizer_steps']}
        history.append(row)
        common.atomic_torch_save(last_path,{'model':model.state_dict(),'architecture':name,'classes':classes,
            'epoch':epoch,'phase':phase,'history':history,'best':best,'config_hash':config_hash,
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict()})
        common.atomic_json(directory/'history.json',history)
        common.write_csv(directory/'history.csv',history,list(row))
        common.atomic_json(directory/'best_validation.json',best)
        print(f'{name} epoch {epoch}: val accuracy={va["accuracy"]:.4f}, macro-F1={va["macro_f1"]:.4f}; best={best["epoch"]}',flush=True)
        previous_phase=phase
    if len(history)!=args.epochs: raise RuntimeError('Every model must complete its full epoch budget')
    common.atomic_json(directory/'training_complete.json',{'epochs':len(history),'best_epoch':best['epoch'],
        'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':sum(r['seconds'] for r in history),
        'best_checkpoint_sha256':common.sha256(directory/'best.pt')})
    del model,optimizer
    gc.collect(); torch.cuda.empty_cache() if args.device=='cuda' else None
    return best

def evaluate_all(args,manifests,classes):
    bests={n:json.loads((args.output/n/'best_validation.json').read_text()) for n in args.models}
    selected=max(args.models,key=lambda n:(bests[n]['macro_f1'],-bests[n]['loss']))
    selection={'selected_model':selected,'criterion':'validation macro-F1, then lower loss',
               'test_used_for_selection':False,'validation_results':bests,'epochs_per_model':args.epochs}
    path=args.output/'selection.json'
    if path.exists() and json.loads(path.read_text())!=selection: raise RuntimeError('Model selection changed after test access')
    common.atomic_json(path,selection)
    print(f'Validation-selected model: {selected}. All training is complete; held-out evaluation begins.',flush=True)
    comparisons=[]
    loaders=loaders_for(args,manifests)
    for name in args.models:
        directory=args.output/name
        metadata=json.loads((directory/'training_complete.json').read_text())
        checkpoint_hash=common.sha256(directory/'best.pt')
        if checkpoint_hash!=metadata['best_checkpoint_sha256']: raise ValueError('Selected checkpoint changed')
        model=Classifier(name,len(classes),pretrained=False).to(args.device)
        model.load_state_dict(common.load_checkpoint(directory/'best.pt')['model'])
        model.set_phase(False)
        for split in ('val','test'):
            marker=directory/f'{split}_evaluation_complete.json'
            if marker.exists():
                if json.loads(marker.read_text())!={'checkpoint_sha256':checkpoint_hash,'samples':len(manifests[split])}:raise ValueError('Evaluation cache changed')
                continue
            metrics,confusion,preds=common.run_epoch(model,loaders[split],torch.device(args.device),len(classes),'benchmark',100,collect=True)
            common.save_evaluation(directory,split,metrics,confusion,preds,manifests[split],classes)
            common.atomic_json(marker,{'checkpoint_sha256':checkpoint_hash,'samples':len(manifests[split])})
        test=json.loads((directory/'test_metrics.json').read_text())
        comparisons.append({'model':name,'selected':name==selected,'epochs':args.epochs,'best_epoch':bests[name]['epoch'],
            **{f'val_{k}':bests[name][k] for k in ('accuracy','macro_f1','loss')},
            **{f'test_{k}':test[k] for k in ('accuracy','macro_f1','loss')},
            'parameters':metadata['parameters'],'training_seconds':metadata['training_seconds']})
        del model; gc.collect(); torch.cuda.empty_cache() if args.device=='cuda' else None
    common.atomic_json(args.output/'comparison.json',comparisons)
    common.write_csv(args.output/'comparison.csv',comparisons,list(comparisons[0]))
    common.atomic_json(args.output/'completed.json',{'selected_model':selected,'epochs_per_model':args.epochs,
                       'pretrained':not args.smoke,'models':args.models})
    print(json.dumps(comparisons,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser()
    for name in ('raw-root','prepared-dir','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--frozen-epochs',type=int,default=5)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    for key in ('raw_root','prepared_dir','output'):setattr(args,key,getattr(args,key).resolve())
    if args.output==args.raw_root or args.raw_root in args.output.parents or args.output==args.prepared_dir or args.prepared_dir in args.output.parents:raise ValueError('Output must be separate from raw data and manifests')
    if not 0<args.frozen_epochs<args.epochs:raise ValueError('Both training phases must have positive lengths')
    if not args.smoke and (args.epochs,args.frozen_epochs,args.seed)!=(20,5,42):raise ValueError('Real benchmark protocol is fixed at 20 epochs, 5 frozen, seed42')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Colab GPU required')
    classes,manifests,hashes=common.read_inputs(args)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    config.update(manifest_sha256=hashes,pretrained_weights=WEIGHTS,
        sources={f:common.sha256(Path(__file__).parents[1]/f) for f in ('train.py','benchmark/train_benchmark.py')},
        classifier='global pooled backbone features -> Linear256 -> LayerNorm -> GELU -> Dropout0.3 -> Linear73',
        transforms={'train':'RGB/EXIF; random resized crop224 area0.7-1,ratio0.75-1.333; flip0.5;colorjitter0.1,0.1,0.1,0.02',
                    'eval':'RGB/EXIF; resize short side256 bilinear antialias; center crop224',
                    'mean':common.MEAN,'std':common.STD},
        optimizer={'name':'AdamW','weight_decay':.01,'frozen_head_lr':1e-3,'finetune_backbone_lr':1e-5,'finetune_head_lr':1e-4,
                   'schedule':'cosine within each phase; reset optimizer at epoch6','clip_norm':1},
        early_stopping=False,stage_transition='continue from epoch5 weights',
        selection='best validation macro-F1 across all20epochs; lower loss breaks ties',
        mixed_precision=str(common.precision_for(args.device)))
    config=json.loads(json.dumps(config))
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if args.resume:
        if json.loads((args.output/'config.json').read_text())!=config:raise ValueError('Resume configuration differs')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        common.atomic_json(args.output/'config.json',config)
        common.atomic_json(args.output/'class_mapping.json',classes)
        common.atomic_json(args.output/'environment.json',common.environment_info(args))
        (args.output/'requirements-lock.txt').write_text('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions() if d.metadata.get('Name'))))
    if (args.output/'completed.json').exists():return
    for name in args.models:fit_one(name,args,manifests,classes,config_hash)
    evaluate_all(args,manifests,classes)

if __name__=='__main__':main()
