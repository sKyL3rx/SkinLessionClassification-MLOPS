from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from lesion_ml.data.dataset import SkinLesionDataset
from lesion_ml.data.transforms import build_transforms_from_config
from lesion_ml.models.factory import build_model_from_config


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader(config: dict[str, Any]) -> DataLoader:
    train_csv = config["data"]["train_csv"]
    val_csv = config["data"]["val_csv"]

    train_tfms = build_transforms_from_config(config, split="train")
    val_tfms = build_transforms_from_config(config, split="val")

    train_ds = SkinLesionDataset(
        csv_path=train_csv,
        transform=train_tfms,
        label_to_idx=None,
        return_metadata=False,
        validate_paths=False,
    )

    label_to_idx = train_ds.label_to_idx

    val_ds = SkinLesionDataset(
        csv_path=val_csv,
        transform=val_tfms,
        label_to_idx=label_to_idx,
        return_metadata=False,
        validate_paths=False,
    )

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, label_to_idx


def build_loss_fn(config: dict[str, Any], label_to_idx: dict[str, int]) -> nn.Module:
    use_class_weights = bool(config["train"].get("class_weights", False))

    if not use_class_weights:
        return nn.CrossEntropyLoss()

    train_csv = config["data"]["train_csv"]
    df = pd.read_csv(train_csv)
    counts = df["label"].astype(str).str.lower().value_counts().to_dict()

    weights = []

    for label_name, _idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        count = counts.get(label_name, 1)
        weights.append(1.0 / max(count, 1))

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)

    return nn.CrossEntropyLoss(weight=weights)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()

    running_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    start_time = time.time()

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    epoch_time = time.time() - start_time

    return epoch_loss, epoch_acc, epoch_f1, epoch_time


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()

    running_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    start_time = time.time()

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        logits = model(images)

        loss = loss_fn(logits, targets)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_macro_f1": best_metric,
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
    device: torch.device,
    epochs: int,
    ckpt_path: Path,
    report_dir: Path,
    config: dict[str, Any],
    label_to_idx: dict[str, int],
) -> dict[str, Any]:
    best_val_f1 = -1.0
    history: list[dict[str, Any]] = []

    early_stopping = bool(config["train"].get("early_stopping", False))
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 5))
    early_stopping_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))

    epochs_without_improvement = 0
    stopped_early = False
    early_stop_epoch = None

    report_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1, train_time = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, val_f1, val_time = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        improved = val_f1 > best_val_f1 + early_stopping_min_delta

        if improved:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_f1,
                config=config,
                label_to_idx=label_to_idx,
            )
            print(f"[INFO] Saved new best checkpoint to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(
                "[INFO] No validation macro-F1 improvement "
                f"for {epochs_without_improvement}/{early_stopping_patience} epoch(s)."
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "train_time_sec": train_time,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "val_time_sec": val_time,
            "best_val_macro_f1_so_far": best_val_f1,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(row)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f} | "
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
        "checkpoint_path": str(ckpt_path),
        "history_csv": str(history_csv_path),
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
    loss_fn = build_loss_fn(config, label_to_idx).to(device)

    optimizer_name = str(config["train"].get("optimizer", "adamw")).lower()
    lr = float(config["train"].get("lr", 3e-4))
    weight_decay = float(config["train"].get("weight_decay", 1e-4))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    epochs = int(config["train"]["epochs"])
    artifact_dir = Path(config["project"].get("artifact_dir", "artifacts"))
    ckpt_path = artifact_dir / "models" / "best.ckpt"
    report_dir = artifact_dir / "reports"

    summary = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        ckpt_path=ckpt_path,
        report_dir=report_dir,
        config=config,
        label_to_idx=label_to_idx,
    )

    print("[INFO] Training finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
