"""Verify real completed results; create scientific plots and an editable report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image,ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'vgg16':'VGG16','resnet50':'ResNet50','mobilenetv2_140':'MobileNetV2 (1.4)','convnext_tiny':'ConvNeXt-Tiny'}
SHORT={'vgg16':'VGG16','resnet50':'ResNet50','mobilenetv2_140':'MobileNetV2','convnext_tiny':'ConvNeXt-Tiny'}
COLORS=['#4679bd','#ce7440','#238c7b','#865fb3']
def read(path):return json.loads(path.read_text())
def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(fig,path):
    fig.savefig(path,dpi=190,bbox_inches='tight',facecolor='white');plt.close(fig)
def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in data]])

def analyze(experiment,raw,out):
    run=experiment/'benchmark_run'; prepared=experiment/'prepared';out.mkdir(parents=True,exist_ok=True)
    plots=out/'plots';plots.mkdir(exist_ok=True)
    config=read(run/'config.json'); complete=read(run/'completed.json'); selection=read(run/'selection.json')
    audit=read(prepared/'audit.json');comparison=read(run/'comparison.json');classes=read(run/'class_mapping.json')
    assert complete['pretrained'] and complete['epochs_per_model']==20 and config['frozen_epochs']==5
    assert set(complete['models'])==set(NAMES) and len(comparison)==4 and audit['status']=='success'
    assert config['seed']==audit['seed']==42 and audit['ratios']=={'train':.8,'val':.1,'test':.1}
    assert len(classes)==73 and audit['retained_images']==7315
    assert audit['split_counts']=={'train':5851,'val':735,'test':729}, 'Report protocol expects the locked finalized split'
    for name,sha in config['manifest_sha256'].items():assert digest(prepared/name)==sha,(name,'manifest changed')
    assert classes==read(prepared/'class_mapping.json')
    test_rows=rows(prepared/'test.csv');truth={r['path']:r for r in test_rows}
    labels={r['object_id']:r['object_name'] for r in classes};ids=list(labels);indices={s:i for i,s in enumerate(ids)}
    all_preds={}; matrices={}; histories={}
    for row in comparison:
        name=row['model'];directory=run/name
        h=read(directory/'history.json');histories[name]=h
        assert [r['epoch'] for r in h]==list(range(1,21))
        assert [r['phase'] for r in h]==['frozen']*5+['finetuned']*15
        assert all(r['optimizer_updates']>0 for r in h)
        best=max(h,key=lambda r:(r['val_macro_f1'],-r['val_loss']))
        assert row['best_epoch']==best['epoch']
        assert abs(row['val_macro_f1']-best['val_macro_f1'])<1e-10
        checkpoint_sha=digest(directory/'best.pt')
        for split in ['val','test']:
            marker=read(directory/f'{split}_evaluation_complete.json')
            assert marker['checkpoint_sha256']==checkpoint_sha and marker['samples']==audit['split_counts'][split]
        predictions=rows(directory/'test_predictions.csv')
        assert len(predictions)==len(truth) and {r['path'] for r in predictions}==set(truth)
        matrix=np.zeros((len(ids),len(ids)),dtype=int)
        for p in predictions:
            expected=truth[p['path']]
            assert p['sha256']==expected['sha256'] and p['true_object_id']==expected['object_id']
            assert p['true_object_name']==labels[p['true_object_id']] and p['predicted_object_name']==labels[p['predicted_object_id']]
            matrix[indices[p['true_object_id']],indices[p['predicted_object_id']]]+=1
        accuracy=np.trace(matrix)/matrix.sum(); tp=matrix.diagonal()
        denominator=matrix.sum(0)+matrix.sum(1)
        macro=float(np.divide(2*tp,denominator,out=np.zeros(len(ids)),where=denominator>0).mean())
        assert abs(accuracy-row['test_accuracy'])<1e-10 and abs(macro-row['test_macro_f1'])<1e-10
        all_preds[name]={p['path']:p for p in predictions};matrices[name]=matrix
    selected=max(comparison,key=lambda r:(r['val_macro_f1'],-r['val_loss']))['model']
    assert selection['selected_model']==complete['selected_model']==selected and selection['test_used_for_selection'] is False
    assert all(r['selected']==(r['model']==selected) for r in comparison)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    x=np.arange(4)
    for ax,metric,title in zip(axes,['accuracy','macro_f1'],['Accuracy','Macro-F1']):
        for offset,split,color in [(-.19,'val','#4679bd'),(.19,'test','#238c7b')]:
            vals=[100*r[f'{split}_{metric}'] for r in comparison]
            bars=ax.bar(x+offset,vals,.36,label='Validation' if split=='val' else 'Test',color=color)
            ax.bar_label(bars,labels=[f'{v:.1f}' for v in vals],fontsize=8,padding=2)
        ax.set(xticks=x,xticklabels=[SHORT[r['model']] for r in comparison],ylim=(0,110),ylabel='Percent',title=title)
        ax.tick_params(axis='x',rotation=15);ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
    save(fig,plots/'comparison.png')
    fig,axes=plt.subplots(4,2,figsize=(10,10),layout='constrained')
    for i,(name,h) in enumerate(histories.items()):
        epochs=[r['epoch'] for r in h]
        best=next(r['best_epoch'] for r in comparison if r['model']==name)
        for ax,metric,title in zip(axes[i],['accuracy','loss'],['Accuracy','Cross-entropy loss']):
            for split,c in [('train','#4679bd'),('val','#ce7440')]:
                ax.plot(epochs,[r[f'{split}_{metric}'] for r in h],color=c,label=split,linewidth=1.8)
            ax.axvline(5.5,color='#777777',linestyle='--',linewidth=.8)
            ax.axvline(best,color='#238c7b',linestyle=':',linewidth=1)
            ax.set(title=f'{NAMES[name]} | {title}',xlabel='Epoch',xlim=(1,20),xticks=[1,5,10,15,20])
            ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
    save(fig,plots/'learning_curves.png')
    matrix=matrices[selected];recall=matrix.diagonal()/matrix.sum(1)
    low=np.argsort(recall,kind='stable')[:10]
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    ax.barh(np.arange(10),100*recall[low],color='#238c7b')
    ax.set(yticks=np.arange(10),yticklabels=[f'{ids[i]} | {textwrap.shorten(labels[ids[i]],width=27)} (n={matrix[i].sum()})' for i in low],xlim=(0,112),xlabel='Test recall (%)',title=f'{NAMES[selected]}: ten lowest class recalls')
    ax.invert_yaxis()
    for y,i in enumerate(low):ax.text(100*recall[i]+1,y,f'{100*recall[i]:.0f}%',va='center',fontsize=9)
    save(fig,plots/'per_class_recall.png')
    fig,ax=plt.subplots(figsize=(17,15),layout='constrained')
    im=ax.imshow(matrix,cmap='Blues');fig.colorbar(im,ax=ax,label='Test images',shrink=.7)
    ax.set(xticks=range(len(ids)),yticks=range(len(ids)),xticklabels=ids,yticklabels=ids,xlabel='Predicted Object ID',ylabel='True Object ID',title=f'{NAMES[selected]}: full test confusion matrix')
    ax.tick_params(axis='x',rotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=7)
    save(fig,plots/'confusion_matrix_full.png')
    errors=sorted([(int(matrix[i,j]),ids[i],ids[j]) for i in range(len(ids)) for j in range(len(ids)) if i!=j and matrix[i,j]],key=lambda z:(-z[0],z[1],z[2]))
    # Fixed random examples are chosen independently of correctness or model confidence.
    fixed=np.random.default_rng(42).choice(len(test_rows),size=min(6,len(test_rows)),replace=False)
    fixed_paths=[test_rows[int(i)]['path'] for i in fixed]
    failure_preds=sorted([p for p in all_preds[selected].values() if p['true_object_id']!=p['predicted_object_id']],key=lambda p:(-float(p['confidence']),p['path']))[:6]
    def display_image(path):
        file=(raw/path).resolve();assert raw.resolve() in file.parents
        assert digest(file)==truth[path]['sha256']
        with Image.open(file) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            w,h=im.size;size=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
            im=im.resize(size,Image.Resampling.BILINEAR)
            left=round((im.width-224)/2);top=round((im.height-224)/2)
            return im.crop((left,top,left+224,top+224))
    def gallery(paths,filename,all_models):
        fig,axes=plt.subplots(2,3,figsize=(10,9.2 if all_models else 7.8))
        for ax in axes.flat:ax.axis('off')
        for ax,path in zip(axes.flat,paths):
            ax.imshow(display_image(path));actual=truth[path]['object_id']
            ax.set_title('\n'.join(textwrap.wrap(f'True: {actual} - {labels[actual]}',32)),fontsize=11,pad=7)
            entries=list(NAMES) if all_models else [selected]
            for i,name in enumerate(entries):
                pr=all_preds[name][path];correct=pr['true_object_id']==pr['predicted_object_id']
                caption=f'{SHORT[name]}: {pr["predicted_object_id"]} ({float(pr["confidence"]):.0%})'
                if not all_models:caption+='\n'+textwrap.shorten(pr['predicted_object_name'],width=30)
                ax.text(.0,-.055-i*.085,caption,transform=ax.transAxes,fontsize=10.5,color='#16734a' if correct else '#b93c3c',va='top')
        fig.subplots_adjust(hspace=.66 if all_models else .43,wspace=.13,top=.91,bottom=.1)
        save(fig,plots/filename)
    gallery(fixed_paths,'shared_examples.png',True)
    gallery([p['path'] for p in failure_preds],'selected_errors.png',False)
    example_selection={'seed':42,'fixed_random_paths':fixed_paths,'error_selection':'six highest-confidence incorrect predictions from selected model; confidence descending then path',
                       'error_paths':[p['path'] for p in failure_preds]}
    (out/'example_selection.json').write_text(json.dumps(example_selection,indent=2)+'\n')
    ordered=sorted(comparison,key=lambda r:(r['val_macro_f1'],-r['val_loss']),reverse=True)
    winner=ordered[0];runner=ordered[1];h=histories[selected]
    smallest=min(comparison,key=lambda r:r['parameters'])
    test_order=sorted(comparison,key=lambda r:r['test_macro_f1'],reverse=True)
    gaps=max(comparison,key=lambda r:r['val_macro_f1']-r['test_macro_f1'])
    ranking_note=(f'{NAMES[gaps["model"]]} has the largest validation-to-test macro-F1 drop '
                  f'({100*(gaps["val_macro_f1"]-gaps["test_macro_f1"]):.2f} percentage points): '
                  f'validation rank {1+next(i for i,r in enumerate(ordered) if r["model"]==gaps["model"])}, '
                  f'test rank {1+next(i for i,r in enumerate(test_order) if r["model"]==gaps["model"])}. '
                  'This ranking change reinforces the limits of a single small test split; it does not alter the validation-based selection.')
    phase_scores={name:{'frozen':max(r['val_macro_f1'] for r in history[:5]),
                        'finetuned':max(r['val_macro_f1'] for r in history[5:])}
                  for name,history in histories.items()}
    background={}
    for name,predictions in all_preds.items():
        background[name]={}
        for key,flag in [('background_tag',True),('other_filenames',False)]:
            subset=[p for p in predictions.values() if ('_bg_' in p['path'].lower())==flag]
            n=len(subset);correct=sum(p['true_object_id']==p['predicted_object_id'] for p in subset)
            background[name][key]={'count':n,'correct':correct,'accuracy':correct/n if n else None}
    bg=background[selected]['background_tag'];other=background[selected]['other_filenames']
    background_note=(f'The test set retains {bg["count"]} filenames containing "_bg_". '
                     f'{NAMES[selected]} classifies {bg["correct"]}/{bg["count"]} correctly ({100*bg["accuracy"]:.2f}%), '
                     f'versus {other["correct"]}/{other["count"]} ({100*other["accuracy"]:.2f}%) among other filenames. '
                     f'Thus {bg["count"]-bg["correct"]} of its {len(test_rows)-int(np.trace(matrix))} errors occur in the filename-tagged background subgroup. '
                     'This is a post-hoc filename-based diagnostic, not manual relabeling; all images remain in the headline metrics.')
    data={'comparison':comparison,'selected':selected,'selected_name':NAMES[selected],
          'selected_correct':int(np.trace(matrix)),'test_count':len(test_rows),'class_count':len(classes),
          'audit':audit,'environment':read(run/'environment.json'),'config':config,'names':NAMES,
          'validation_margin_pp':100*(winner['val_macro_f1']-runner['val_macro_f1']),
          'runner_up':NAMES[runner['model']], 'histories':histories,'phase_scores':phase_scores,'smallest_model':smallest,'ranking_note':ranking_note,
          'top_errors':[{'count':n,'true_id':t,'true_name':labels[t],'predicted_id':p,'predicted_name':labels[p]} for n,t,p in errors[:8]],
          'support_min':int(matrix.sum(1).min()),'support_max':int(matrix.sum(1).max()),
          'examples':example_selection,'experiment':str(experiment.resolve()),'plots':str(plots.resolve()),
          'background_diagnostic':background,'background_note':background_note,
          'checks':'All20epochs/model,5+15phasecounts,allmanifesthashes,checkpoint-evaluationhashes,allpredictionpaths/IDs,accuracyandmacroF1recomputedfrompredictions'}
    (out/'report_data.json').write_text(json.dumps(data,indent=2)+'\n')
    md=['# Milestone 1: a controlled four-model CNN comparison',
        '## Dataset and classification task',
        'The task is 73-class object recognition using the finalized merged dataset and official Object IDs. The locked shared dataset was not changed. Validation and exact duplicate detection produced 7,315 retained images: 5,851 training, 735 validation and 729 test images (80/10/10 targets, seed 42, per-class integer rounding).',
        'All 7,343 images decoded successfully, including 410 JPEG/MPO primary images. 28 exact duplicates were excluded from manifests and 2 metadata files ignored. Raw files were preserved; folders were not capped at 100 images and background images were not removed by filename. Byte and oriented RGB-pixel hashes are disjoint across splits.',
        '## Why the original results were not directly comparable',
        'The supplied scripts use VGG16, ResNet50 and MobileNetV2 (width 1.4); the previous experiment used ConvNeXt-Tiny. Split construction, image exclusions, heads, augmentation, hyperparameter searches and stopping rules differed. This benchmark reruns all four under one protocol; the original files and their old scores are preserved and are not pooled into this comparison.',
        'The implementations use Torchvision VGG16, ResNet50 (V1.5), ConvNeXt-Tiny and timm MobileNetV2-1.4. All start from ImageNet-1K weights. These are standardized adaptations of the model families, not byte-for-byte re-executions of the Keras notebooks: the ResNet variant and MobileNet pretrained checkpoint differ. Different pretrained recipes remain part of the comparison.',
        '## Common preprocessing and experimental protocol',
        'Each model receives the identical saved split manifests and official class mapping. Images use EXIF orientation and RGB. Training uses random resized 224 x 224 crops (area 70-100%, aspect ratio 3/4-4/3), horizontal flip probability 0.5 and mild color jitter (brightness/contrast/saturation 0.1, hue 0.02). Evaluation resizes the short side to 256 and center-crops to 224 x 224. All tensors use ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). Only training is augmented.',
        'All backbones feed globally pooled features into the same head: Linear(256), LayerNorm, GELU, Dropout(0.3), Linear(73). Feature dimensions vary by architecture. Each model trains for exactly 20 epochs: 5 with frozen backbone/evaluation mode, then 15 with the entire model trainable. Fine-tuning continues from epoch 5; no early stopping or per-model search is used. AdamW uses weight decay 0.01, batch size 32, gradient clipping 1.0, and cosine decay within each phase. Head learning rate is 0.001 when frozen; fine-tuning uses 0.00001 for the backbone and 0.0001 for the head. The optimizer resets at phase transition. Model/data RNGs use seed 42 and reproducible epoch-derived seeds; batch order and augmentation streams are shared across models.',
        f'Execution used Colab CLI on {data["environment"]["gpu"]} with {config["mixed_precision"]}. Checkpoints and final model selection use validation macro-F1, with lower validation loss breaking ties. All four training runs finished before test evaluation began. The test split was reused from the earlier ConvNeXt pilot; it is a common fixed benchmark rather than a new independent test collection.',
        '## Results',
        table(['Model','Epochs','Best epoch','Val accuracy','Val macro-F1','Test accuracy','Test macro-F1'],[[NAMES[r['model']],20,r['best_epoch'],f'{100*r["val_accuracy"]:.2f}%',f'{r["val_macro_f1"]:.4f}',f'{100*r["test_accuracy"]:.2f}%',f'{r["test_macro_f1"]:.4f}'] for r in comparison]),
        '![Validation and test comparison](plots/comparison.png)',
        table(['Model','Parameters (M)','Epoch time total (min)'],[[NAMES[r['model']],f'{r["parameters"]/1e6:.2f}',f'{r["training_seconds"]/60:.2f}'] for r in comparison]),
        'Timing includes training, validation and best-checkpoint saving within each epoch; it excludes setup, weight downloads and export. These wall times depend on the input pipeline and are not an isolated GPU or inference-speed benchmark. Equal epochs do not imply equal compute cost.',
        '## Comparison, convergence and final model selection',
        f'**{NAMES[selected]}** is selected at epoch **{winner["best_epoch"]}**: its validation macro-F1 is {winner["val_macro_f1"]:.4f}, a {data["validation_margin_pp"]:.2f} percentage-point margin over {NAMES[runner["model"]]}. Its test accuracy is {100*winner["test_accuracy"]:.2f}% ({int(np.trace(matrix))}/{len(test_rows)} correct), and test macro-F1 is {winner["test_macro_f1"]:.4f}. The decision follows validation ranking, even if test ranking differs.',
        f'{NAMES[smallest["model"]]} is the smallest tested configuration at {smallest["parameters"]/1e6:.2f} million parameters, with test macro-F1 {smallest["test_macro_f1"]:.4f}. The selected model has {winner["parameters"]/1e6:.2f} million parameters. This documents the accuracy-size tradeoff; no inference-latency or deployment benchmark was performed.',
        ranking_note,
        f'For the selected model, epoch 20 training/validation accuracies are {100*h[-1]["train_accuracy"]:.2f}%/{100*h[-1]["val_accuracy"]:.2f}%. Best-epoch validation macro-F1 is {winner["val_macro_f1"]:.4f}, versus {h[-1]["val_macro_f1"]:.4f} at epoch 20. Training curves use augmented inputs and changing weights; validation uses fixed crops, so their gap is descriptive rather than a controlled overfitting estimate.',
        table(['Model','Best frozen val F1','Best fine-tuned val F1','Fine-tuning gain (pp)'],
              [[NAMES[name],f'{v["frozen"]:.4f}',f'{v["finetuned"]:.4f}',f'{100*(v["finetuned"]-v["frozen"]):+.2f}'] for name,v in phase_scores.items()]),
        'The phase comparison uses validation scores only. The phases are sequential, with unequal lengths (5 and 15 epochs), so the gain reflects both unfreezing and additional training rather than an isolated causal effect of unfreezing.',
        '![All 20-epoch learning curves](plots/learning_curves.png)',
        'Dashed line marks the transition after epoch 5; dotted line marks each selected checkpoint. Accuracy is on a 0-1 scale; loss is cross-entropy.',
        '## Per-class errors',
        '![Lowest class recalls for selected model](plots/per_class_recall.png)',
        table(['True Object ID/name','Predicted Object ID/name','Images'],[[f'{e["true_id"]}: {e["true_name"]}',f'{e["predicted_id"]}: {e["predicted_name"]}',e['count']] for e in data['top_errors']]),
        'The table lists directed confusion counts, sorted by count then official ID. The full 73-class confusion matrix is available in plots/confusion_matrix_full.png and model prediction/per-class CSVs are in the experiment artifacts.',
        background_note,
        table(['Model','Accuracy on _bg_ filenames (n=38)','Accuracy on other filenames (n=691)'],
              [[NAMES[name],f'{100*v["background_tag"]["accuracy"]:.2f}%',f'{100*v["other_filenames"]["accuracy"]:.2f}%'] for name,v in background.items()]),
        'This subgroup analysis uses saved predictions and does not change model selection, training, labels or the test denominator. The supplied MobileNet notebook excludes these files by name; retaining them in the common pipeline is one reason its earlier numbers cannot be compared directly.',
        '## Shared prediction examples',
        'Six test paths are selected with NumPy seed 42 independently of predictions. The same evaluation crop is shown once, with predictions from all four models. Green labels are correct and red labels are incorrect; percentages are uncalibrated softmax confidence.',
        '![Same test examples across all four models](plots/shared_examples.png)',
        '## Failure examples',
        'These are up to six highest-confidence mistakes from the validation-selected model, ordered by confidence then path. They are deliberately selected failures, not a random estimate of error frequency. Captions use official IDs and names.',
        '![Selected model mistakes](plots/selected_errors.png)',
        'In the displayed crops, the plush-doll-labelled background image prominently shows a bag, and the portable-SSD-labelled background image shows a bowl. Other examples show bedding or room context without an obvious target. These observations suggest ambiguity or context sensitivity in some retained samples; they do not establish that the official labels should be changed. The locked labels and all images were preserved.',
        '## Limitations and reproducibility',
        f'Test support is only {data["support_min"]}-{data["support_max"]} images per class. Exact deduplication does not remove near-duplicate views or shared backgrounds; image-level splitting may retain capture correlations. One split/seed and different ImageNet pretraining recipes do not establish a universal architecture ranking or statistical significance. Shared hyperparameters improve comparability but are not necessarily optimal for each backbone. Further validation should use independently captured objects/scenes. Cross-hardware bitwise reproducibility is not guaranteed.',
        'The archived run includes manifests, official IDs, exclusions, configuration, environment/package lock, all 20-epoch histories, best and final/resumable checkpoints, per-class metrics, confusion matrices and per-image predictions. benchmark/train_benchmark.py implements training; benchmark/run_colab_benchmark.py launches and retrieves the run. example_selection.json records the exact illustrated paths. The report generator verifies manifests, completed epoch budgets, checkpoint hashes and prediction-derived metrics before rendering.',
        '## Sources',
        '- [Final dataset](https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS)\n- [Final Object IDs](https://docs.google.com/spreadsheets/d/1OCBWlTK8JSzrgUxON3thJWAKRKU2Pi-V9Ssn2BbDxds/edit#gid=0)\n- [Torchvision model weights](https://docs.pytorch.org/vision/stable/models.html)\n- [MobileNetV2-1.4 weight card](https://huggingface.co/timm/mobilenetv2_140.ra_in1k)']
    text='\n\n'.join(md)+'\n'
    (out/'group_comparison_report.md').write_text(text)
    print(json.dumps({'selected':selected,'plots':str(plots),'report':str(out/'group_comparison_report.md')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.experiment,a.raw_root,a.output)
