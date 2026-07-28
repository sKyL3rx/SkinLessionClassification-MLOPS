from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from lesion_ml.data.dataset import SkinLesionDataset
from lesion_ml.data.metadata import MetadataSchema, build_metadata_schema
from lesion_ml.data.transforms import build_transforms_from_config
from lesion_ml.models.factory import build_model_from_config
from lesion_ml.models.forward import forward_batch
from lesion_ml.models.losses import FocalLoss
from lesion_ml.paths import get_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a skin lesion classification model.")
    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def resume_model_weights(
    model: nn.Module,
    resume_from: str | Path | None,
    device: torch.device,
    strict: bool = True,
) -> dict[str, Any] | None:

    if not resume_from:
        return None

    checkpoint = load_checkpoint(resume_from, device=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {resume_from}")

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    if not strict:
        print(f"[INFO] Resume missing_keys: {missing_keys}")
        print(f"[INFO] Resume unexpected_keys: {unexpected_keys}")

    print(
        f"[INFO] Resumed model weights from: {resume_from} "
        f"(epoch={checkpoint.get('epoch')}, "
        f"best_val_macro_f1={checkpoint.get('best_val_macro_f1')}, "
        f"checkpoint_source={checkpoint.get('checkpoint_source', 'unknown')})"
    )

    return checkpoint


def resume_optimizer_if_requested(
    optimizer: torch.optim.Optimizer,
    checkpoint: dict[str, Any] | None,
    config: dict[str, Any],
) -> None:
    if checkpoint is None:
        return

    if not bool(config["train"].get("resume_optimizer", False)):
        return

    if "optimizer_state_dict" not in checkpoint:
        raise KeyError("resume_optimizer=True but checkpoint has no optimizer_state_dict")

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("[INFO] Resumed optimizer state from checkpoint.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:

        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")

        self.ema = deepcopy(model).eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()

            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def build_weighted_sampler(
    train_ds: SkinLesionDataset,
    label_to_idx: dict[str, int],
) -> WeightedRandomSampler:
    labels = train_ds.df["label"].astype(str).str.lower().tolist()
    label_indices = [label_to_idx[label] for label in labels]

    counts = np.bincount(label_indices, minlength=len(label_to_idx))
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = np.array([class_weights[idx] for idx in label_indices], dtype=np.float64)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloader(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_csv = config["data"]["train_csv"]
    val_csv = config["data"]["val_csv"]

    train_tfms = build_transforms_from_config(config, split="train")
    val_tfms = build_transforms_from_config(config, split="val")

    use_metadata = bool(config["train"].get("use_metadata", False))
    metadata_schema: MetadataSchema | None = None

    preprocess_cfg = config.get("preprocess", {})
    preprocess_mode = str(preprocess_cfg.get("mode", "none"))
    lesion_crop_margin = float(preprocess_cfg.get("lesion_crop_margin", 0.20))
    dark_border_threshold = int(preprocess_cfg.get("dark_border_threshold", 10))

    print(
        "[INFO] Preprocess: "
        f"mode={preprocess_mode}, "
        f"resize_mode={preprocess_cfg.get('resize_mode', 'resize_pad')}, "
        f"lesion_crop_margin={lesion_crop_margin}, "
        f"dark_border_threshold={dark_border_threshold}"
    )

    if use_metadata:
        metadata_schema = build_metadata_schema(train_csv)
        config["train"]["metadata_dim"] = metadata_schema.dim
        print(f"[INFO] Metadata fusion enabled. metadata_dim={metadata_schema.dim}")

    train_ds = SkinLesionDataset(
        csv_path=train_csv,
        transform=train_tfms,
        label_to_idx=None,
        return_metadata=use_metadata,
        metadata_schema=metadata_schema,
        validate_paths=False,
        preprocess_mode=preprocess_mode,
        lesion_crop_margin=lesion_crop_margin,
        dark_border_threshold=dark_border_threshold,
    )

    label_to_idx = train_ds.label_to_idx

    val_ds = SkinLesionDataset(
        csv_path=val_csv,
        transform=val_tfms,
        label_to_idx=label_to_idx,
        return_metadata=use_metadata,
        metadata_schema=metadata_schema,
        validate_paths=False,
        preprocess_mode=preprocess_mode,
        lesion_crop_margin=lesion_crop_margin,
        dark_border_threshold=dark_border_threshold,
    )

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 0))

    use_weighted_sampler = bool(config["train"].get("weighted_sampler", False))
    sampler = build_weighted_sampler(train_ds, label_to_idx) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, label_to_idx


def compute_inverse_class_weights(
    train_csv: str,
    label_to_idx: dict[str, int],
) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df["label"].astype(str).str.lower().value_counts().to_dict()

    weights = []
    for label_name, _idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        count = counts.get(label_name, 1)
        weights.append(1.0 / max(count, 1))

    weights_t = torch.tensor(weights, dtype=torch.float32)
    weights_t = weights_t / weights_t.sum() * len(weights_t)

    return weights_t


def compute_effective_num_weights(
    train_csv: str,
    label_to_idx: dict[str, int],
    beta: float = 0.999,
) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts_dict = df["label"].astype(str).str.lower().value_counts().to_dict()

    counts = []
    for label_name, _idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        counts.append(max(int(counts_dict.get(label_name, 1)), 1))

    counts_t = torch.tensor(counts, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)

    effective_num = 1.0 - torch.pow(beta_t, counts_t)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(weights)

    return weights


def build_loss_fn(config: dict[str, Any], label_to_idx: dict[str, int]) -> nn.Module:
    train_cfg = config["train"]

    loss_name = str(train_cfg.get("loss", "cross_entropy")).lower()
    train_csv = config["data"]["train_csv"]

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    use_class_weights = bool(train_cfg.get("class_weights", False))

    weight = None

    if loss_name in {"cb_focal", "class_balanced_focal"}:
        beta = float(train_cfg.get("cb_beta", 0.999))
        gamma = float(train_cfg.get("focal_gamma", 2.0))

        weight = compute_effective_num_weights(
            train_csv=train_csv,
            label_to_idx=label_to_idx,
            beta=beta,
        )

        return FocalLoss(
            gamma=gamma,
            weight=weight,
            label_smoothing=label_smoothing,
        )

    if use_class_weights:
        weight = compute_inverse_class_weights(
            train_csv=train_csv,
            label_to_idx=label_to_idx,
        )

    if loss_name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )

    if loss_name == "focal":
        gamma = float(train_cfg.get("focal_gamma", 2.0))

        return FocalLoss(
            gamma=gamma,
            weight=weight,
            label_smoothing=label_smoothing,
        )

    raise ValueError(f"Unsupported loss: {loss_name}")


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["train"]

    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    base_lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    backbone_lr_mult = float(train_cfg.get("backbone_lr_mult", 1.0))
    head_lr_mult = float(train_cfg.get("head_lr_mult", 1.0))

    has_metadata_wrapper = hasattr(model, "image_backbone")

    head_prefixes = (
        "metadata_mlp.",
        "image_projection.",
        "metadata_projection.",
        "gate.",
        "classifier.",
        "head.",
        "fc.",
    )

    param_groups = {
        "backbone_decay": {
            "params": [],
            "lr": base_lr * backbone_lr_mult,
            "weight_decay": weight_decay,
        },
        "backbone_no_decay": {
            "params": [],
            "lr": base_lr * backbone_lr_mult,
            "weight_decay": 0.0,
        },
        "head_decay": {
            "params": [],
            "lr": base_lr * head_lr_mult,
            "weight_decay": weight_decay,
        },
        "head_no_decay": {
            "params": [],
            "lr": base_lr * head_lr_mult,
            "weight_decay": 0.0,
        },
    }

    def is_no_decay_param(param_name: str, param: torch.nn.Parameter) -> bool:
        name_l = param_name.lower()
        return (
            param.ndim <= 1
            or name_l.endswith(".bias")
            or ".norm" in name_l
            or "norm." in name_l
            or ".bn" in name_l
            or "bn." in name_l
            or "layernorm" in name_l
            or "batchnorm" in name_l
        )

    def is_head_param(param_name: str) -> bool:
        name_l = param_name.lower()

        if has_metadata_wrapper:
            # MetadataFusionClassifier:
            # image_backbone.* = pretrained visual backbone
            # everything else = metadata/fusion/classifier head
            return not name_l.startswith("image_backbone.")

        return name_l.startswith(head_prefixes)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_decay = is_no_decay_param(name, param)
        is_head = is_head_param(name)

        if is_head and no_decay:
            param_groups["head_no_decay"]["params"].append(param)
        elif is_head:
            param_groups["head_decay"]["params"].append(param)
        elif no_decay:
            param_groups["backbone_no_decay"]["params"].append(param)
        else:
            param_groups["backbone_decay"]["params"].append(param)

    groups = []
    for group_name, group in param_groups.items():
        if group["params"]:
            group["name"] = group_name
            groups.append(group)

    if not groups:
        raise ValueError("No trainable parameters found for optimizer.")

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            groups,
            betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
            eps=float(train_cfg.get("eps", 1e-8)),
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            groups,
            betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
            eps=float(train_cfg.get("eps", 1e-8)),
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            groups,
            momentum=float(train_cfg.get("momentum", 0.9)),
            nesterov=bool(train_cfg.get("nesterov", True)),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    steps_per_epoch: int,
) -> LambdaLR | None:
    train_cfg = config["train"]

    scheduler_name = str(train_cfg.get("scheduler", "none")).lower()

    if scheduler_name in {"none", "null", ""}:
        return None

    if scheduler_name not in {"cosine", "cosine_warmup"}:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg.get("lr", 3e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))

    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = max(warmup_epochs * steps_per_epoch, 0)
    min_lr_ratio = min_lr / lr

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)

        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: LambdaLR | None,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    ema_model: ModelEMA | None = None,
) -> tuple[float, float, float, float, float]:
    model.train()

    running_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    use_amp = bool(config["train"].get("mixed_precision", False)) and device.type == "cuda"
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))
    channels_last = bool(config["train"].get("channels_last", False)) and device.type == "cuda"

    start_time = time.time()

    max_batches = config["train"].get("max_train_batches")
    max_batches = int(max_batches) if max_batches not in (None, "", 0, "0") else None
    samples_seen = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
            batch = dict(batch)
            batch["image"] = images

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = forward_batch(
                model=model,
                batch=batch,
                device=device,
            )
            loss = loss_fn(logits, targets)

        if scaler.is_enabled():
            scaler.scale(loss).backward()

            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()

            optimizer_stepped = new_scale >= old_scale

        else:
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            optimizer_stepped = True

        if optimizer_stepped:
            if ema_model is not None:
                ema_model.update(model)

            if scheduler is not None:
                scheduler.step()

        batch_size = images.size(0)
        samples_seen += batch_size
        running_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        if max_batches is not None and batch_idx >= max_batches:
            break

    epoch_loss = running_loss / max(samples_seen, 1)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    epoch_time = time.time() - start_time
    current_lr = max(group["lr"] for group in optimizer.param_groups)

    return epoch_loss, epoch_acc, epoch_f1, epoch_time, current_lr


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, float, float]:
    model.eval()

    running_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    start_time = time.time()
    samples_seen = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        batch = dict(batch)
        batch["image"] = images

        logits = forward_batch(
            model=model,
            batch=batch,
            device=device,
        )
        loss = loss_fn(logits, targets)

        batch_size = images.size(0)
        samples_seen += batch_size
        running_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        if max_batches is not None and batch_idx >= max_batches:
            break

    epoch_loss = running_loss / max(samples_seen, 1)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    epoch_time = time.time() - start_time

    return epoch_loss, epoch_acc, epoch_f1, epoch_time


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    label_to_idx: dict[str, int],
    checkpoint_source: str = "raw",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_macro_f1": best_metric,
        "checkpoint_source": checkpoint_source,
        "model_ema": checkpoint_source == "ema",
        "config": config,
        "label_to_idx": label_to_idx,
    }
    torch.save(checkpoint, path)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epochs: int,
    ckpt_path: Path,
    report_dir: Path,
    config: dict[str, Any],
    label_to_idx: dict[str, int],
    ema_model: ModelEMA | None = None,
) -> dict[str, Any]:
    best_val_f1 = -1.0
    history: list[dict[str, Any]] = []

    best_checkpoint_source = "none"

    early_stopping = bool(config["train"].get("early_stopping", False))
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 5))
    early_stopping_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    max_val_batches = config["train"].get("max_val_batches")
    max_val_batches = int(max_val_batches) if max_val_batches not in (None, "", 0, "0") else None

    epochs_without_improvement = 0
    stopped_early = False
    early_stop_epoch = None

    report_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1, train_time, current_lr = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            ema_model=ema_model,
        )

        val_loss, val_acc, val_f1, val_time = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            max_batches=max_val_batches,
        )

        ema_val_loss = None
        ema_val_acc = None
        ema_val_f1 = None
        ema_val_time = None

        # Default: use raw model for checkpoint selection.
        selection_f1 = val_f1
        selection_model = model
        selection_source = "raw"

        if ema_model is not None:
            ema_val_loss, ema_val_acc, ema_val_f1, ema_val_time = evaluate(
                model=ema_model.ema,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                max_batches=max_val_batches,
            )

            if ema_val_f1 > val_f1:
                selection_f1 = ema_val_f1
                selection_model = ema_model.ema
                selection_source = "ema"

        improved = selection_f1 > best_val_f1 + early_stopping_min_delta

        if improved:
            best_val_f1 = selection_f1
            epochs_without_improvement = 0

            best_checkpoint_source = selection_source

            save_checkpoint(
                path=ckpt_path,
                model=selection_model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_f1,
                config=config,
                label_to_idx=label_to_idx,
                checkpoint_source=selection_source,
            )
            print(
                f"[INFO] Saved new best checkpoint to {ckpt_path} "
                f"using {selection_source} model "
                f"(selection_val_macro_f1={selection_f1:.4f})"
            )
        else:
            epochs_without_improvement += 1
            print(
                "[INFO] No validation macro-F1 improvement "
                f"for {epochs_without_improvement}/{early_stopping_patience} epoch(s)."
            )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "train_time_sec": train_time,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "val_time_sec": val_time,
            "ema_val_loss": ema_val_loss,
            "ema_val_acc": ema_val_acc,
            "ema_val_macro_f1": ema_val_f1,
            "ema_val_time_sec": ema_val_time,
            "selection_val_macro_f1": selection_f1,
            "selection_source": selection_source,
            "best_val_macro_f1_so_far": best_val_f1,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(row)

        if ema_model is not None:
            ema_msg = f" | ema_val_f1={ema_val_f1:.4f}"
        else:
            ema_msg = ""

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"lr={current_lr:.8f} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
            f"{ema_msg} | "
            f"selection={selection_source} "
            f"best_val_f1={best_val_f1:.4f}"
        )

        if early_stopping and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            early_stop_epoch = epoch
            print(
                f"[INFO] Early stopping triggered at epoch {epoch}. "
                f"Best val_macro_f1={best_val_f1:.4f}"
            )
            break

    history_df = pd.DataFrame(history)
    history_csv_path = report_dir / "train_history.csv"
    history_df.to_csv(history_csv_path, index=False)

    summary = {
        "best_val_macro_f1": best_val_f1,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "early_stop_epoch": early_stop_epoch,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "backbone": config["train"]["backbone"],
        "scheduler": str(config["train"].get("scheduler", "none")),
        "warmup_epochs": int(config["train"].get("warmup_epochs", 0)),
        "min_lr": float(config["train"].get("min_lr", 0.0)),
        "model_ema": ema_model is not None,
        "model_ema_decay": float(config["train"].get("model_ema_decay", 0.0)),
        "test_used_for_model_selection": False,
        "loss": str(config["train"].get("loss", "cross_entropy")),
        "class_weights": bool(config["train"].get("class_weights", False)),
        "weighted_sampler": bool(config["train"].get("weighted_sampler", False)),
        "label_smoothing": float(config["train"].get("label_smoothing", 0.0)),
        "mixed_precision": bool(config["train"].get("mixed_precision", False)),
        "channels_last": bool(config["train"].get("channels_last", False)),
        "grad_clip_norm": float(config["train"].get("grad_clip_norm", 0.0)),
        "checkpoint_path": str(ckpt_path),
        "use_metadata": bool(config["train"].get("use_metadata", False)),
        "metadata_dim": config["train"].get("metadata_dim"),
        "metadata_hidden_dim": int(config["train"].get("metadata_hidden_dim", 64)),
        "fusion_hidden_dim": int(config["train"].get("fusion_hidden_dim", 256)),
        "history_csv": str(history_csv_path),
        "resume_from": config["train"].get("resume_from"),
        "resume_optimizer": bool(config["train"].get("resume_optimizer", False)),
        "resume_strict": bool(config["train"].get("resume_strict", True)),
        "selection_metric": "max(raw_val_macro_f1, ema_val_macro_f1)"
        if ema_model is not None
        else "val_macro_f1",
        "checkpoint_source": best_checkpoint_source,
        "fusion_type": str(config["train"].get("fusion_type", "concat")),
    }

    summary_json_path = report_dir / "train_metrics.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["project"].get("seed", 42))
    set_seed(seed)

    device = get_device()
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader, label_to_idx = build_dataloader(config)

    model = build_model_from_config(config).to(device)

    if bool(config["train"].get("channels_last", False)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    resume_from = config["train"].get("resume_from")
    resume_strict = bool(config["train"].get("resume_strict", True))

    resume_checkpoint = resume_model_weights(
        model=model,
        resume_from=resume_from,
        device=device,
        strict=resume_strict,
    )

    ema_model = None
    if bool(config["train"].get("model_ema", False)):
        ema_model = ModelEMA(
            model=model,
            decay=float(config["train"].get("model_ema_decay", 0.9998)),
        )
        print(
            "[INFO] Model EMA enabled. "
            f"decay={float(config['train'].get('model_ema_decay', 0.9998))}"
        )

    loss_fn = build_loss_fn(config, label_to_idx).to(device)
    optimizer = build_optimizer(model, config)

    resume_optimizer_if_requested(
        optimizer=optimizer,
        checkpoint=resume_checkpoint,
        config=config,
    )

    max_train_batches = config["train"].get("max_train_batches")
    max_train_batches = (
        int(max_train_batches) if max_train_batches not in (None, "", 0, "0") else None
    )
    steps_per_epoch = len(train_loader)
    if max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_train_batches)

    scheduler = build_scheduler(
        optimizer=optimizer,
        config=config,
        steps_per_epoch=steps_per_epoch,
    )

    use_amp = bool(config["train"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(config["train"]["epochs"])

    paths = get_project_paths(config)
    run_dir = paths.run_dir
    ckpt_path = paths.checkpoint_path
    report_dir = paths.reports_dir

    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Checkpoint path: {ckpt_path}")
    print(f"[INFO] Report directory: {report_dir}")

    summary = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        epochs=epochs,
        ckpt_path=ckpt_path,
        report_dir=report_dir,
        config=config,
        label_to_idx=label_to_idx,
        ema_model=ema_model,
    )

    print("[INFO] Training finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
