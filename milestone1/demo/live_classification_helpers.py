"""ConvNeXt-Tiny inference matching the Milestone 1 controlled benchmark."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import textwrap
import uuid

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageDraw, ImageFont, ImageOps

MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def temporary_cwd(directory):
    original = Path.cwd()
    Path(directory).mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original)


def upload_to(directory):
    from google.colab import files
    directory = Path(directory).resolve()
    with temporary_cwd(directory):
        uploaded = files.upload()
    paths = []
    for name in uploaded:
        path = (directory / name).resolve()
        if directory not in path.parents:
            raise ValueError('Unsafe upload filename')
        paths.append(path)
    return paths


def verify_model_file(path, expected_sha256):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Model file not found: {path}')
    if sha256_file(path) != expected_sha256:
        raise ValueError('This is not our selected classification checkpoint. '
                         'Upload milestone1_convnext_tiny_best.pt supplied with this notebook.')
    return path


class LiveClassifier(nn.Module):
    """Exact ConvNeXt branch and common classifier head used in training."""
    def __init__(self, classes=73):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()
        self.head = nn.Sequential(nn.Linear(features, 256), nn.LayerNorm(256),
                                  nn.GELU(), nn.Dropout(.3), nn.Linear(256, classes))

    def forward(self, images):
        return self.head(self.backbone(images))


def load_classifier(path, expected_sha256, mapping, device='cpu'):
    path = verify_model_file(path, expected_sha256)
    # Read only the exact trusted checkpoint whose hash was verified above.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != 'convnext_tiny' or checkpoint.get('epoch') != 15:
        raise ValueError('Checkpoint architecture/epoch differs from the selected experiment')
    if checkpoint.get('classes') != mapping or len(mapping) != 73:
        raise ValueError('Checkpoint classes differ from the final Object IDs/names')
    if [m['class_index'] for m in mapping] != list(range(73)):
        raise ValueError('Invalid final class mapping')
    model = LiveClassifier(len(mapping))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    return model


def evaluation_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def result_card(image, prediction):
    width = max(600, min(1000, image.width))
    preview = ImageOps.contain(image, (width, 650), method=Image.Resampling.BILINEAR)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 26)
        small = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError:
        font, small = ImageFont.load_default(size=26), ImageFont.load_default(size=20)
    lines = textwrap.wrap(prediction['object_name'], width=max(20, int((width-32)/16)))
    header = 104 + 34*len(lines)
    panel = Image.new('RGB', (width, header+preview.height+16), 'white')
    draw = ImageDraw.Draw(panel)
    draw.text((16, 12), f"Predicted Object ID: {prediction['object_id']}", font=font, fill='#005c61')
    for index, line in enumerate(lines):
        draw.text((16, 48+34*index), line, font=font, fill='#162f46')
    draw.text((16, header-36), f"Model score: {prediction['score']:.2%}", font=small, fill='#516474')
    panel.paste(preview, ((width-preview.width)//2, header))
    return panel


def classify_image(model, image_path, mapping, output, device='cpu', model_sha256=None, top_k=5):
    image_path, output = Path(image_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as original:
        original.seek(0)
        image = ImageOps.exif_transpose(original).convert('RGB')
    tensor = evaluation_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = model(tensor).softmax(1)[0].cpu()
    if scores.numel() != len(mapping) or not torch.isfinite(scores).all():
        raise ValueError('Invalid classifier output')
    values, indices = scores.topk(min(top_k, len(mapping)))
    ranked = [{'rank': rank, 'object_id': mapping[index]['object_id'],
               'object_name': mapping[index]['object_name'], 'score': float(score)}
              for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), 1)]
    record = {'input_filename': image_path.name, 'input_sha256': sha256_file(image_path),
              'model_sha256': model_sha256, 'architecture': 'convnext_tiny', 'selected_epoch': 15,
              'width': image.width, 'height': image.height, 'prediction': ranked[0], 'top_k': ranked,
              'score_type': 'softmax model score; not a calibrated correctness probability',
              'preprocessing': {'orientation': 'EXIF-oriented primary RGB photo',
                                'resize_short_side': 256, 'center_crop': [224, 224],
                                'mean': list(MEAN), 'std': list(STD)},
              'generated_utc': datetime.now(timezone.utc).isoformat()}
    prefix = image_path.stem+'_'+uuid.uuid4().hex[:8]
    card = output/f'{prefix}_classification.png'
    result_card(image, ranked[0]).save(card)
    (output/f'{prefix}_classification.json').write_text(json.dumps(record, indent=2))
    with (output/f'{prefix}_classification.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', 'object_id', 'object_name', 'score'])
        writer.writeheader()
        writer.writerows(ranked)
    return record, ranked, card
