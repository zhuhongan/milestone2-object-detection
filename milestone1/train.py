#!/usr/bin/env python3
"""Reproducible ConvNeXt-Tiny transfer learning on immutable split manifests.

Run with CUDA for real experiments. ``--weights none --device cpu`` is intended
only for smoke tests; its outputs are explicitly labelled as non-pretrained.
Only this run's output directory is modified; source images remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be set before a CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
STAGES = ("frozen", "finetuned")
REQUIRED_COLUMNS = {"path", "object_id", "object_name", "class_index", "sha256", "pixel_sha256"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True, type=Path)
    p.add_argument("--prepared-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--frozen-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--frozen-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--weights", choices=("IMAGENET1K_V1", "none"), default="IMAGENET1K_V1")
    p.add_argument("--resume", action="store_true", help="Resume this output directory from completed epochs, with identical settings.")
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args(argv)
    for field in ("batch_size", "frozen_epochs", "finetune_epochs", "patience", "log_every"):
        if getattr(args, field) < 1:
            p.error(f"--{field.replace('_', '-')} must be positive")
    if args.workers < 0 or args.seed < 0 or args.seed >= 2**32:
        p.error("workers must be nonnegative and seed must be in [0, 2**32)")
    if min(args.frozen_lr, args.backbone_lr, args.head_lr) <= 0 or args.weight_decay < 0:
        p.error("learning rates must be positive and weight decay nonnegative")
    for field in ("raw_root", "prepared_dir", "output"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.output == args.raw_root or args.raw_root in args.output.parents:
        p.error("Output must be outside the locked raw dataset")
    if args.output == args.prepared_dir or args.prepared_dir in args.output.parents:
        p.error("Output must be separate from prepared manifests")
    return args


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_torch_save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def write_csv(path, rows, fields):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_inputs(args):
    audit_path = args.prepared_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "success":
        raise ValueError("Preprocessing audit is not successful; resolve it before training")
    if audit.get("seed") != args.seed or audit.get("ratios") != {"train": 0.8, "val": 0.1, "test": 0.1}:
        raise ValueError("Preprocessing must use this run's seed and the requested 80/10/10 ratios")
    mapping_path = args.prepared_dir / "class_mapping.json"
    mapping_digest = sha256(mapping_path)
    if mapping_digest != audit.get("class_mapping_sha256"):
        raise ValueError("Class mapping changed after the successful preprocessing audit")
    classes = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("class_mapping.json must contain at least two classes")
    classes = sorted(classes, key=lambda row: int(row["class_index"]))
    if [int(row["class_index"]) for row in classes] != list(range(len(classes))):
        raise ValueError("class_index must be contiguous and start at zero")
    if len({str(row["object_id"]) for row in classes}) != len(classes):
        raise ValueError("Each official Object ID must identify exactly one class")
    manifests, hashes = {}, {"class_mapping.json": mapping_digest, "audit.json": sha256(audit_path)}
    observed = {field: {} for field in ("path", "sha256", "pixel_sha256")}
    for split in ("train", "val", "test"):
        path = args.prepared_dir / f"{split}.csv"
        hashes[path.name] = sha256(path)
        if hashes[path.name] != audit.get("manifest_sha256", {}).get(path.name):
            raise ValueError(f"{path.name} changed after the successful preprocessing audit")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
                raise ValueError(f"{path.name} needs columns {sorted(REQUIRED_COLUMNS)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{split} manifest is empty")
        for row in rows:
            index = int(row["class_index"])
            if index < 0 or index >= len(classes):
                raise ValueError(f"Invalid class_index: {row}")
            label = classes[index]
            if row["object_id"] != str(label["object_id"]) or row["object_name"] != label["object_name"]:
                raise ValueError(f"Object ID/name does not match class mapping: {row['path']}")
            relative = Path(row["path"])
            image_path = (args.raw_root / relative).resolve()
            if relative.is_absolute() or args.raw_root not in image_path.parents or not image_path.is_file():
                raise ValueError(f"Manifest image is missing or outside raw-root: {row['path']}")
            for field in observed:
                value = row[field]
                if not value:
                    raise ValueError(f"Empty {field} in {split}: {row['path']}")
                if value in observed[field]:
                    raise ValueError(f"Duplicate {field}: {row['path']} ({split}) and {observed[field][value]}")
                observed[field][value] = f"{split}/{row['path']}"
            if sha256(image_path) != row["sha256"]:
                raise ValueError(f"Raw image changed after preparation: {row['path']}")
            row["class_index"] = index
        if {r["class_index"] for r in rows} != set(range(len(classes))):
            raise ValueError(f"Every class must occur in {split}; regenerate valid stratified manifests")
        manifests[split] = rows
        print(f"Verified {split}: {len(rows)} immutable source images", flush=True)
    return classes, manifests, hashes


class ManifestDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Input has already passed the preparation audit. Size alone must not
        # cause validated high-resolution images to be silently excluded.
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(self.root / row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = self.transform(image)
        return tensor, row["class_index"], index


def make_loaders(args, manifests):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_transform = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    generators = {split: torch.Generator().manual_seed(args.seed + offset) for offset, split in enumerate(manifests)}
    loaders = {}
    for split, rows in manifests.items():
        dataset = ManifestDataset(args.raw_root, rows, train_transform if split == "train" else eval_transform)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                    num_workers=args.workers, pin_memory=args.device == "cuda",
                                    worker_init_fn=seed_worker, generator=generators[split],
                                    persistent_workers=False, drop_last=False)
    return loaders, generators


def build_model(class_count, weights):
    pretrained = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if weights == "IMAGENET1K_V1" else None
    model = convnext_tiny(weights=pretrained)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, class_count)
    return model


def configure_stage(model, stage, args):
    for param in model.parameters():
        param.requires_grad_(stage == "finetuned")
    for param in model.classifier[2].parameters():
        param.requires_grad_(True)
    if stage == "frozen":
        groups = [{"params": model.classifier[2].parameters(), "lr": args.frozen_lr}]
    else:
        head_ids = {id(p) for p in model.classifier[2].parameters()}
        groups = [
            {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.backbone_lr},
            {"params": model.classifier[2].parameters(), "lr": args.head_lr},
        ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def metrics_from_confusion(confusion, loss):
    matrix = confusion.astype(np.float64)
    tp = matrix.diagonal()
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return {"loss": float(loss), "accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "per_class": [{"precision": float(precision[i]), "recall": float(recall[i]),
                           "f1": float(f1[i]), "support": int(support[i])} for i in range(len(tp))]}


def precision_for(device):
    """BF16 avoids FP16's limited exponent range on supported GPUs."""
    if torch.device(device).type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def run_epoch(model, loader, device, class_count, stage, log_every, optimizer=None, scaler=None, collect=False):
    training = optimizer is not None
    if training and stage == "frozen":
        model.eval()  # Frozen features have no random stochastic-depth masks.
        model.classifier[2].train()
    else:
        model.train(training)
    total_loss, total_seen = 0.0, 0
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    predictions = []
    optimizer_updates, skipped_optimizer_steps = 0, 0
    precision = precision_for(device)
    started = time.monotonic()
    for step, (images, labels, indices) in enumerate(loader, start=1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=precision is not None, dtype=precision):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite loss; experiment stopped without silently dropping images")
        if training:
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                                    error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            # GradScaler lowers its scale and skips optimizer.step on overflow.
            # A stable/growing scale (or disabled scaling) means a real update.
            if scaler.get_scale() < previous_scale:
                skipped_optimizer_steps += 1
                print(f"  {stage} skipped optimizer update at batch {step} after FP16 overflow; scale={scaler.get_scale():g}", flush=True)
            else:
                optimizer_updates += 1
        guessed = logits.detach().argmax(dim=1)
        y, pred = labels.cpu().numpy(), guessed.cpu().numpy()
        np.add.at(confusion, (y, pred), 1)
        batch_n = len(y)
        total_loss += float(loss.detach()) * batch_n
        total_seen += batch_n
        if collect:
            confidence = logits.detach().float().softmax(dim=1).max(dim=1).values.cpu().numpy()
            predictions.extend({"row_index": int(index), "true_class_index": int(actual),
                                "predicted_class_index": int(guess), "confidence": float(prob)}
                               for index, actual, guess, prob in zip(indices.tolist(), y, pred, confidence))
        if step == 1 or step % log_every == 0 or step == len(loader):
            phase = "train" if training else "evaluate"
            print(f"  {stage} {phase} {step}/{len(loader)} loss={total_loss / total_seen:.4f} elapsed={time.monotonic() - started:.1f}s", flush=True)
    metrics = metrics_from_confusion(confusion, total_loss / total_seen)
    if training:
        if optimizer_updates == 0:
            raise RuntimeError("No optimizer updates completed this epoch; refusing to save an untrained checkpoint. Mixed-precision gradients overflowed on every batch.")
        metrics.update(optimizer_updates=optimizer_updates, skipped_optimizer_steps=skipped_optimizer_steps)
    return metrics, confusion, predictions


def rng_state(generators):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
             "loaders": {name: generator.get_state() for name, generator in generators.items()}}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, generators):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    for name, value in state["loaders"].items():
        generators[name].set_state(value.cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def load_checkpoint(path):
    # Only locally generated checkpoints from this explicitly chosen run are read.
    return torch.load(path, map_location="cpu", weights_only=False)


def better(candidate, best):
    return best is None or (candidate["macro_f1"], -candidate["loss"]) > (best["macro_f1"], -best["loss"])


def fit_stage(model, stage, args, loaders, generators, classes, config_hash):
    directory = args.output / stage
    directory.mkdir(exist_ok=True)
    best_path, last_path = directory / "best.pt", directory / "last.pt"
    epochs = args.frozen_epochs if stage == "frozen" else args.finetune_epochs
    optimizer = configure_stage(model, stage, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=precision_for(args.device) == torch.float16, init_scale=1024.0)
    history, best, bad_epochs, start = [], None, 0, 1
    if args.resume and last_path.exists():
        state = load_checkpoint(last_path)
        if state["config_hash"] != config_hash:
            raise ValueError("Resume checkpoint configuration does not match this run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Optimizer state is copied to each parameter's device by load_state_dict.
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best, bad_epochs, start = state["history"], state["best"], state["bad_epochs"], state["epoch"] + 1
        restore_rng(state["rng"], generators)
        print(f"Resuming {stage} from completed epoch {state['epoch']}", flush=True)
    elif stage == "finetuned":
        source = load_checkpoint(args.output / "frozen" / "best.pt")
        model.load_state_dict(source["model"])
        # The second stage uses a reproducible, independent augmentation stream.
        seed_everything(args.seed + 1)
        for offset, generator in enumerate(generators.values()):
            generator.manual_seed(args.seed + 100 + offset)
    for epoch in range(start, epochs + 1):
        if bad_epochs >= args.patience:
            break
        began = time.monotonic()
        lrs = [group["lr"] for group in optimizer.param_groups]
        print(f"{stage}: epoch {epoch}/{epochs}; learning rates={lrs}", flush=True)
        train_metrics, _, _ = run_epoch(model, loaders["train"], torch.device(args.device), len(classes), stage,
                                        args.log_every, optimizer=optimizer, scaler=scaler)
        val_metrics, _, _ = run_epoch(model, loaders["val"], torch.device(args.device), len(classes), stage, args.log_every)
        improved = better(val_metrics, best)
        if improved:
            best = dict(val_metrics, epoch=epoch)
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
               "train_macro_f1": train_metrics["macro_f1"], "val_loss": val_metrics["loss"],
               "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
               "optimizer_updates": train_metrics["optimizer_updates"], "skipped_optimizer_steps": train_metrics["skipped_optimizer_steps"],
               "backbone_or_frozen_lr": lrs[0], "head_lr": lrs[-1], "seconds": time.monotonic() - began}
        history.append(row)
        scheduler.step()
        state = {"stage": stage, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
                 "best": best, "bad_epochs": bad_epochs, "rng": rng_state(generators),
                 "config_hash": config_hash, "classes": classes, "architecture": "convnext_tiny", "weights": args.weights}
        if improved:
            atomic_torch_save(best_path, {k: v for k, v in state.items() if k not in ("optimizer", "scheduler", "scaler", "rng")})
        atomic_torch_save(last_path, state)
        write_csv(directory / "history.csv", history, list(row))
        atomic_json(directory / "history.json", history)
        atomic_json(directory / "best_validation.json", best)
        print(f"{stage} epoch {epoch}: val accuracy={val_metrics['accuracy']:.4f}, macro-F1={val_metrics['macro_f1']:.4f}, loss={val_metrics['loss']:.4f}; best epoch={best['epoch']}", flush=True)
    if best is None or not best_path.exists():
        raise RuntimeError(f"{stage} has no valid completed checkpoint")
    # Repair lightweight summaries if an interruption followed the checkpoint.
    write_csv(directory / "history.csv", history, list(history[0]))
    atomic_json(directory / "history.json", history)
    atomic_json(directory / "best_validation.json", best)
    return best


def save_evaluation(directory, split, metrics, confusion, predictions, rows, classes):
    atomic_json(directory / f"{split}_metrics.json", metrics)
    per_class = [dict(classes[index], **result) for index, result in enumerate(metrics["per_class"])]
    write_csv(directory / f"{split}_per_class.csv", per_class, list(per_class[0]))
    ids = [str(row["object_id"]) for row in classes]
    confusion_rows = [dict(true_object_id=ids[i], **{f"predicted_{key}": int(confusion[i, j]) for j, key in enumerate(ids)}) for i in range(len(ids))]
    write_csv(directory / f"{split}_confusion_matrix.csv", confusion_rows, list(confusion_rows[0]))
    output_rows = []
    for prediction in predictions:
        actual = classes[prediction["true_class_index"]]
        guessed = classes[prediction["predicted_class_index"]]
        original = rows[prediction["row_index"]]
        output_rows.append({"path": original["path"], "sha256": original["sha256"],
                            "true_object_id": actual["object_id"], "true_object_name": actual["object_name"],
                            "predicted_object_id": guessed["object_id"], "predicted_object_name": guessed["object_name"],
                            "confidence": prediction["confidence"], "correct": actual["class_index"] == guessed["class_index"]})
    write_csv(directory / f"{split}_predictions.csv", output_rows, list(output_rows[0]))


def draw_curves(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for index, stage in enumerate(STAGES):
        history = json.loads((output / stage / "history.json").read_text())
        epochs = [row["epoch"] for row in history]
        for ax, metric in zip(axes[index], ("loss", "accuracy", "macro_f1")):
            for split in ("train", "val"):
                ax.plot(epochs, [row[f"{split}_{metric}"] for row in history], marker="o", label=split)
            ax.set(title=f"{stage}: {metric.replace('_', ' ')}", xlabel="Epoch")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)


def environment_info(args):
    info = {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "mixed_precision": str(precision_for(args.device)),
            "deterministic_algorithms": True,
            "reproducibility_limit": "Identical results are not guaranteed across hardware, PyTorch releases, or platform changes."}
    return info


def initialize_output(args, classes, manifest_hashes):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "resume"}
    config.update(architecture="convnext_tiny", stages=list(STAGES), manifest_sha256=manifest_hashes,
                  mixed_precision=str(precision_for(args.device)), fp16_initial_scale=1024.0,
                  training_source_sha256=sha256(Path(__file__)),
                  model_selection="Highest validation macro-F1; ties resolved by lower validation cross-entropy loss. Test never selects a model.",
                  transforms={"train": "EXIF transpose, RGB, RandomResizedCrop224 scale0.7-1 ratio3/4-4/3 bilinear antialias; horizontal flip p0.5; brightness/contrast/saturation jitter0.1, hue0.02; ToTensor; ImageNet normalization",
                              "eval": "EXIF transpose, RGB, resize short side236 bilinear antialias, center crop224, ToTensor, ImageNet mean/std"})
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.resume:
        if not (args.output / "config.json").is_file():
            raise ValueError("--resume requires an existing run with config.json")
        previous = json.loads((args.output / "config.json").read_text())
        if previous != config:
            raise ValueError("Run settings or manifests changed; create a new --output directory")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / "config.json", config)
        atomic_json(args.output / "class_mapping.json", classes)
        atomic_json(args.output / "environment.json", environment_info(args))
        requirements = []
        for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata.get("Name", "").lower()):
            name = distribution.metadata.get("Name")
            if name:
                requirements.append(f"{name}=={distribution.version}")
        (args.output / "requirements-lock.txt").write_text("\n".join(requirements) + "\n")
        shutil.copy2(__file__, args.output / "train_source.py")
        manifest_dir = args.output / "manifests"
        manifest_dir.mkdir()
        for filename in manifest_hashes:
            shutil.copy2(args.prepared_dir / filename, manifest_dir / filename)
    return config_hash


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the requested experiment. Attach a Colab GPU runtime, or explicitly use --device cpu only for a smoke test.")
    if args.weights == "none":
        print("SMOKE/UNPRETRAINED RUN: ImageNet weights disabled; these are not pretrained experiment results.", flush=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output already exists. Choose a new run name or use --resume with unchanged settings.")
    seed_everything(args.seed)
    classes, manifests, manifest_hashes = read_inputs(args)
    config_hash = initialize_output(args, classes, manifest_hashes)
    if (args.output / "completed.json").exists():
        print(f"Run already complete: {args.output / 'comparison.csv'}", flush=True)
        return
    loaders, generators = make_loaders(args, manifests)
    # Resuming never downloads a second copy of pretrained weights.
    have_checkpoint = args.resume and (args.output / "frozen" / "last.pt").exists()
    model = build_model(len(classes), "none" if have_checkpoint else args.weights).to(args.device)
    bests = {stage: fit_stage(model, stage, args, loaders, generators, classes, config_hash) for stage in STAGES}
    selected = max(STAGES, key=lambda stage: (bests[stage]["macro_f1"], -bests[stage]["loss"]))
    # Persist selection before loading any test pixels.
    atomic_json(args.output / "selection.json", {"selected_stage": selected, "criterion": "validation macro-F1, then lower validation loss",
                                               "validation_results": bests, "test_used_for_selection": False,
                                               "checkpoint": f"{selected}/best.pt", "pretrained": args.weights != "none"})
    print(f"Selected from validation: {selected}; now evaluating the held-out test set for both fixed models.", flush=True)
    comparisons = []
    for stage in STAGES:
        directory = args.output / stage
        state = load_checkpoint(directory / "best.pt")
        model.load_state_dict(state["model"])
        # Each evaluation completion marker prevents repeated test access on resume.
        for split in ("val", "test"):
            marker = directory / f"{split}_evaluation_complete.json"
            if not marker.exists():
                metrics, confusion, predictions = run_epoch(model, loaders[split], torch.device(args.device), len(classes), stage, args.log_every, collect=True)
                save_evaluation(directory, split, metrics, confusion, predictions, manifests[split], classes)
                atomic_json(marker, {"checkpoint_sha256": sha256(directory / "best.pt"), "samples": len(manifests[split])})
        test = json.loads((directory / "test_metrics.json").read_text())
        comparisons.append({"approach": stage, "pretrained": args.weights != "none", "selected_from_validation": stage == selected,
                            "best_epoch": bests[stage]["epoch"], "validation_accuracy": bests[stage]["accuracy"],
                            "validation_macro_f1": bests[stage]["macro_f1"], "validation_loss": bests[stage]["loss"],
                            "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"], "test_loss": test["loss"]})
    write_csv(args.output / "comparison.csv", comparisons, list(comparisons[0]))
    atomic_json(args.output / "comparison.json", comparisons)
    draw_curves(args.output)
    atomic_json(args.output / "completed.json", {"utc": datetime.now(timezone.utc).isoformat(), "selected_stage": selected,
                                               "selected_checkpoint": f"{selected}/best.pt", "pretrained": args.weights != "none"})
    print(json.dumps(comparisons, indent=2), flush=True)
    print(f"Completed. Results and official Object IDs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
