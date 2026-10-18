"""Create the six-page report from verified report_data.json and actual figures."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from PIL import Image as PILImage

NAVY=colors.HexColor('#162F46');TEAL=colors.HexColor('#087E83');GRAY=colors.HexColor('#516474')
PAGE=(612,792);MARGIN=44;WIDTH=PAGE[0]-2*MARGIN

def make_pdf(source,output):
    output.parent.mkdir(parents=True,exist_ok=True)
    d=json.loads(source.read_text());out=source.parent;plots=out/'plots';selected=d['selected']
    results=d['comparison'];names=d['names'];winner=next(r for r in results if r['model']==selected)
    history=d['histories'][selected];audit=d['audit']
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=24,leading=28,textColor=NAVY,spaceAfter=12))
    styles.add(ParagraphStyle(name='SubtitleCustom',fontName='Helvetica',fontSize=11,leading=15,textColor=GRAY,spaceAfter=15))
    styles.add(ParagraphStyle(name='SectionCustom',fontName='Helvetica-Bold',fontSize=14,leading=18,textColor=NAVY,spaceBefore=8,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.5,leading=13.3,textColor=NAVY,spaceAfter=7))
    styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8,leading=10.5,textColor=GRAY,spaceAfter=6))
    styles.add(ParagraphStyle(name='CellCustom',fontName='Helvetica',fontSize=8.2,leading=10.5,textColor=NAVY))
    styles.add(ParagraphStyle(name='HeaderCell',fontName='Helvetica-Bold',fontSize=8.3,leading=10.8,textColor=colors.white))
    story=[]
    def P(s,style='BodyCustom'):return Paragraph(s,styles[style])
    def add(s,style='BodyCustom'):story.append(P(s,style))
    def title(n,t):
        add(f'MILESTONE 1  /  {n:02d}','SmallCustom');add(t,'TitleCustom')
    def section(t):add(t,'SectionCustom')
    def table(headers,data,widths):
        cells=[[P(escape(str(v)),'HeaderCell') for v in headers]]+[[P(escape(str(v)),'CellCustom') for v in r] for r in data]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EDF4F5'),colors.white]),
            ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#B7C8D0'))]))
        story.extend([t,Spacer(1,9)])
    def image(name,max_height):
        path=plots/name
        with PILImage.open(path) as im:w,h=im.size
        scale=min(WIDTH/w,max_height/h)
        story.append(Image(str(path),width=w*scale,height=h*scale,hAlign='CENTER'))
    def new():story.append(PageBreak())

    title(1,'Controlled comparison<br/>of four CNN models')
    add('Controlled object-classification comparison | 20 epochs per model | Seed 42','SubtitleCustom')
    add(f'<b>Selected model: {escape(d["selected_name"])}.</b> Validation macro-F1: <b>{winner["val_macro_f1"]:.4f}</b>. '
        f'Held-out test accuracy: <b>{100*winner["test_accuracy"]:.2f}%</b>; test macro-F1: <b>{winner["test_macro_f1"]:.4f}</b>. '
        'Selection used validation results before this benchmark accessed test predictions.')
    section('Dataset and problem')
    add('Classify photographs into <b>73 official Object IDs</b> from the finalized merged dataset. The shared data and Object ID Sheet remain unchanged. '
        'Class indices are internal model targets; all exported predictions and error analyses use the official IDs and names.')
    table(['Audit','Count','Split','Images'],[
        ['Valid images','7,343','Training','5,851'],['Exact duplicates excluded','28','Validation','735'],
        ['Retained images','7,315','Test','729'],['Metadata files ignored','2','Object classes','73']], [182,58,155,129])
    add('The requested 80/10/10 allocation is stratified by class using seed 42 and largest-remainder integer rounding. '
        'Every model reuses the exact same saved manifests. Per-class integer counts explain the slight departure from exact global percentages. '
        'Each test class has 9-11 images.','SmallCustom')
    section('Validation and preprocessing')
    add('All images decoded successfully, including 410 JPEG/MPO containers handled using primary frame 0. '
        'EXIF orientation and RGB conversion are applied in memory. Exact file hashes and oriented RGB-pixel hashes were checked before splitting; '
        'same-label duplicate copies were excluded from manifests. Raw files were preserved. No folder was capped at 100 images and no background images were dropped by filename.')
    add('Training: random resized 224 x 224 crop (70-100% area, aspect ratio 3/4-4/3), 50% horizontal flip and mild color jitter '
        '(brightness/contrast/saturation 0.1; hue 0.02). Validation/test: resize shorter side to 256, then center crop to 224 x 224. '
        'Bilinear antialiasing and ImageNet RGB normalization are shared by all models. Augmentation applies only to training.')
    section('Why rerun the group models?')
    add('The supplied scripts differed in split logic, exclusions, classifier heads, augmentation, tuning budgets and stopping rules. '
        'This report compares a fresh common protocol, rather than mixing their previously reported scores.')
    add('Data: drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS<br/>'
        'Object IDs: docs.google.com/spreadsheets/d/1OCBWlTK8JSzrgUxON3thJWAKRKU2Pi-V9Ssn2BbDxds','SmallCustom')
    new()

    title(2,'Models, training and results')
    add('Four ImageNet-1K backbones: VGG16, ResNet50, MobileNetV2 (width 1.4) and ConvNeXt-Tiny. '
        'Each uses globally pooled features and the same classifier: Linear(256), LayerNorm, GELU, Dropout(0.3), Linear(73). '
        'Backbone feature widths differ; classifier structure is shared.')
    table(['Common setting','Protocol'],[
        ['Epoch budget','Exactly 20: 5 frozen + 15 full fine-tuning; no early stopping'],
        ['Optimization','AdamW; batch 32; weight decay 0.01; gradient clipping 1.0'],
        ['Learning rates','Frozen head: 0.001. Fine-tuning backbone/head: 0.00001 / 0.0001'],
        ['Schedule and transition','Cosine per phase; reset optimizer at epoch 6; continue epoch-5 weights'],
        ['Selection','Highest validation macro-F1; lower validation loss breaks ties'],
        ['Hardware',f'Colab CLI / {d["environment"]["gpu"]} / BF16 mixed precision']], [115,409])
    add('All models use the same seed-derived epoch shuffles and augmentation streams. Frozen backbones remain in evaluation mode; '
        'all layers, including batch-normalization layers where present, become trainable during fine-tuning.','SmallCustom')
    table(['Model','Best epoch','Val macro-F1','Test acc.','Test macro-F1'],[
        [names[r['model']]+(' *' if r['selected'] else ''),r['best_epoch'],f'{r["val_macro_f1"]:.4f}',f'{100*r["test_accuracy"]:.2f}%',f'{r["test_macro_f1"]:.4f}'] for r in results], [182,70,90,92,90])
    add('* Validation-selected model. Every model completed 20 epochs; each row evaluates its best validation checkpoint.','SmallCustom')
    image('comparison.png',155)
    add('Implementations are standardized adaptations: Torchvision VGG16/ResNet50 V1.5/ConvNeXt-Tiny and timm MobileNetV2-1.4. '
        'The ResNet variant and MobileNet pretrained weights differ from the original Keras notebooks. Common preprocessing includes a 256-pixel evaluation resize for all backbones. '
        'Thus these results compare model-and-weight configurations under one downstream protocol, not architecture alone.','SmallCustom')
    new()

    title(3,'Learning across all 20 epochs')
    add('Blue: augmented training. Orange: fixed-crop validation. Dashed gray line: unfreezing after epoch 5. '
        'Dotted green line: validation-selected checkpoint. Accuracy uses a 0-1 scale.','SmallCustom')
    image('learning_curves.png',512)
    add(f'For {escape(d["selected_name"])}, the selected checkpoint is epoch {winner["best_epoch"]}. '
        f'At epoch 20, training and validation accuracy are {100*history[-1]["train_accuracy"]:.2f}% and {100*history[-1]["val_accuracy"]:.2f}%; '
        f'validation macro-F1 is {history[-1]["val_macro_f1"]:.4f}, compared with {winner["val_macro_f1"]:.4f} at the selected checkpoint. '
        'Training metrics use stochastic augmentation and changing weights, so the train-validation gap is descriptive rather than a controlled overfitting estimate.')
    new()

    title(4,'Comparison and error analysis')
    add(f'<b>{escape(d["selected_name"])}</b> leads the validation ranking by <b>{d["validation_margin_pp"]:.2f} macro-F1 percentage points</b> '
        f'over {escape(d["runner_up"])}. It correctly classifies <b>{d["selected_correct"]} of {d["test_count"]}</b> test images. '
        'Macro-F1 gives equal weight to each class and complements overall accuracy.')
    phase=d['phase_scores'][selected]
    add(f'Its best validation macro-F1 is {phase["frozen"]:.4f} in the frozen phase and {phase["finetuned"]:.4f} during fine-tuning. '
        'The phases differ in duration, so the change reflects unfreezing plus additional training.','SmallCustom')
    add(escape(d['ranking_note']),'SmallCustom')
    add(escape(d['background_note']),'SmallCustom')
    table(['Model','Parameters (M)','Epoch time total (min)'],[
        [names[r['model']],f'{r["parameters"]/1e6:.2f}',f'{r["training_seconds"]/60:.2f}'] for r in results], [232,132,160])
    add('Time includes training, validation and best-checkpoint saving inside each epoch; setup, weight downloads and export are excluded. '
        'Wall time depends on the input pipeline; this is not an isolated GPU or inference-speed benchmark. Equal epochs do not mean equal compute cost.','SmallCustom')
    image('per_class_recall.png',207)
    table(['True Object ID / name','Predicted Object ID / name','Count'],[
        [f'{e["true_id"]} / {e["true_name"]}',f'{e["predicted_id"]} / {e["predicted_name"]}',e['count']] for e in d['top_errors'][:5]], [238,238,48])
    add('Directed errors are sorted by count, then official ID. Small per-class support makes one error a large recall change. '
        'The full 73-class confusion matrix and complete prediction CSVs accompany this report.','SmallCustom')
    new()

    title(5,'Same images, four predictions')
    add('Six fixed test examples were sampled with NumPy seed 42 independently of model predictions. Each image shows the actual shared evaluation crop. '
        'Green labels are correct; red labels are incorrect. Percentages are softmax confidence, not calibrated correctness probabilities.','SmallCustom')
    image('shared_examples.png',546)
    add('These six illustrations are not the basis for the reported accuracy. Metrics use every one of the 729 test images. '
        'Exact paths and the sampling rule are saved in example_selection.json.','SmallCustom')
    new()

    title(6,'Failure cases and final assessment')
    add(f'Up to six highest-confidence mistakes from {escape(d["selected_name"])} are shown below, sorted by confidence then path. '
        'This deliberately selected failure gallery illustrates errors; it does not estimate their frequency.','SmallCustom')
    image('selected_errors.png',330)
    add('Two retained background samples show a bag under the Plush doll label and a bowl under the Portable SSD label. '
        'Other crops show room context without an obvious target. These observations flag ambiguity; official labels and all samples were preserved.','SmallCustom')
    section('Final choice and limits')
    add(f'Select <b>{escape(d["selected_name"])} at epoch {winner["best_epoch"]}</b> because it has the strongest validation macro-F1 under the agreed budget. '
        f'Its test accuracy is {100*winner["test_accuracy"]:.2f}% and macro-F1 is {winner["test_macro_f1"]:.4f}. '
        'The choice was fixed before this benchmark evaluated the test set.')
    small=d['smallest_model']
    add(f'{escape(names[small["model"]])} is the smallest configuration ({small["parameters"]/1e6:.2f}M parameters; test macro-F1 {small["test_macro_f1"]:.4f}), '
        f'while the selected model has {winner["parameters"]/1e6:.2f}M parameters. This is an accuracy-size comparison, not an inference-latency test.','SmallCustom')
    add('This is one image-level split with 9-11 test images per class. Exact deduplication does not remove near-duplicate views or shared backgrounds. '
        'The test split was used in the earlier ConvNeXt pilot, so this is a reused fixed benchmark, not a fresh independent test collection. '
        'Pretraining recipes differ and shared hyperparameters need not be optimal for every model. No universal architecture ranking or statistical significance is claimed. '
        'An independent capture-based test set would provide a stronger generalization check.','SmallCustom')
    add('Reproducibility: all 80 model-epochs, split and checkpoint hashes, official IDs, environment, source, histories, per-image predictions and resumable checkpoints '
        'are archived. Run benchmark/run_colab_benchmark.py to reproduce the protocol. Original group scripts and shared raw data are unchanged.','SmallCustom')
    add('Model references: <link href="https://docs.pytorch.org/vision/stable/models.html" color="#087E83">Torchvision models and weights</link>; '
        '<link href="https://huggingface.co/timm/mobilenetv2_140.ra_in1k" color="#087E83">timm MobileNetV2-1.4 weight card</link>. '
        'See the editable Markdown report and archived config for exact weight identifiers and full dataset links.','SmallCustom')

    output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=PAGE,rightMargin=MARGIN,leftMargin=MARGIN,topMargin=36,bottomMargin=36,
                          title='Milestone 1 - Controlled Four-Model CNN Comparison',author='Project group')
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#CBD6DE'));canvas.line(MARGIN,28,PAGE[0]-MARGIN,28)
        canvas.setFont('Helvetica',8);canvas.setFillColor(GRAY)
        canvas.drawString(MARGIN,16,'Milestone 1 | Final merged dataset | Seed 42')
        canvas.drawRightString(PAGE[0]-MARGIN,16,f'{doc.page}')
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();make_pdf(a.data,a.output)
