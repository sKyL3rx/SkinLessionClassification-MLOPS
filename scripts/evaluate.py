from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from lesion_ml.data.dataset import SkinLesionDataset
from lesion_ml.data.transforms import build_transforms_from_config
from lesion_ml.models.factory import build_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained skin lesion model.")
    parser.add_argument("--config", type=str, default="params.yaml")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help = "Data split to evaluate"
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("inference", {}).get("device", "cpu")).lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def build_test_dataloader(
    config: dict[str, Any],
    label_to_idx: dict[str, int],
) -> DataLoader:
    test_csv = Path(config["data"]["test_csv"])

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    test_dataset = SkinLesionDataset(
        csv_path=test_csv,
        transform=build_transforms_from_config(config, split="test"),
        label_to_idx=label_to_idx,
        return_metadata=False,
        validate_paths=False,
    )

    return DataLoader(
        test_dataset,
        batch_size=int(config["train"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

def build_eval_dataloader(
        config: dict[str,Any],
        label_to_idx: dict[str, int],
        split: str
) -> DataLoader:
    split_to_csv_key = {
        "train": "train_csv",
        "val": "val_csv",
        "test": "test_csv",
    }

    csv_key = split_to_csv_key[split]
    csv_path = Path(config["data"][csv_key])

    if not csv_path.exists():
        raise FileNotFoundError(f"{split} CSV not found: {csv_path}")
    
    dataset = SkinLesionDataset(
        csv_path=csv_path,
        transform=build_transforms_from_config(config, split="test"),
        label_to_idx=label_to_idx,
        return_metadata=False,
        validate_paths=False,
    )

    return DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

def get_tta_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    eval_cfg = config.get("evaluate", {})

    use_tta = bool(eval_cfg.get("use_tta", False))
    tta_transforms = eval_cfg.get("tta_transforms", ["original"])

    if not isinstance(tta_transforms, list) or len(tta_transforms) == 0:
        tta_transforms = ["original"]

    tta_transforms = [str(name).lower() for name in tta_transforms]

    if "original" not in tta_transforms:
        tta_transforms = ["original", *tta_transforms]

    if not use_tta:
        tta_transforms = ["original"]

    return use_tta, tta_transforms


def apply_tta_transform(images: torch.Tensor, transform_name: str) -> torch.Tensor:
    transform_name = transform_name.lower()

    if transform_name == "original":
        return images

    if transform_name in {"hflip", "horizontal_flip"}:
        return torch.flip(images, dims=[3])

    if transform_name in {"vflip", "vertical_flip"}:
        return torch.flip(images, dims=[2])

    if transform_name in {"hvflip", "hflip_vflip", "vflip_hflip"}:
        return torch.flip(images, dims=[2, 3])

    raise ValueError(f"Unsupported TTA transform: {transform_name}")


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module,
    images: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    use_tta, tta_transforms = get_tta_config(config)

    if not use_tta:
        logits = model(images)
        return torch.softmax(logits, dim=1)

    probs_sum: torch.Tensor | None = None

    for transform_name in tta_transforms:
        images_tta = apply_tta_transform(images, transform_name)
        logits = model(images_tta)
        probs = torch.softmax(logits, dim=1)

        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum = probs_sum + probs

    if probs_sum is None:
        raise RuntimeError("TTA produced no probabilities.")

    return probs_sum / float(len(tta_transforms))


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()

    all_probs: list[np.ndarray] = []
    all_targets: list[int] = []
    all_image_ids: list[str] = []
    all_image_paths: list[str] = []

    use_tta, tta_transforms = get_tta_config(config)

    if use_tta:
        print(f"[INFO] TTA enabled with transforms: {tta_transforms}")
    else:
        print("[INFO] TTA disabled.")

    start_time = time.time()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].detach().cpu().numpy()

        probs = predict_proba(
            model=model,
            images=images,
            config=config,
        ).detach().cpu().numpy()

        all_probs.append(probs)
        all_targets.extend(targets.tolist())
        all_image_ids.extend(list(batch["image_id"]))
        all_image_paths.extend(list(batch["image_path"]))

    if not all_probs:
        raise RuntimeError("No predictions were generated. Check the test dataloader.")

    dur_time = time.time() - start_time
    print(f"[INFO] Test inference finished in: {dur_time:.2f}s")

    return (
        np.concatenate(all_probs, axis=0),
        np.array(all_targets),
        all_image_ids,
        all_image_paths,
    )


def save_eval_artifacts(
    report_dir: Path,
    probs: np.ndarray,
    targets: np.ndarray,
    image_ids: list[str],
    image_paths: list[str],
    idx_to_label: dict[int, str],
    top_k: int,
    confidence_threshold: float,
    model_metadata: dict[str, Any],
    eval_metadata: dict[str, Any],
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)

    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    class_indices = list(range(len(idx_to_label)))
    class_names = [idx_to_label[i] for i in class_indices]

    accuracy = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    balanced_acc = balanced_accuracy_score(targets, preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=class_indices,
        zero_division=0,
    )

    per_class_df = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    per_class_path = report_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)

    cm = confusion_matrix(targets, preds, labels=class_indices)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = report_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    top_k = min(top_k, probs.shape[1])
    top_indices = np.argsort(-probs, axis=1)[:, :top_k]

    prediction_rows = []

    for i in range(len(targets)):
        row = {
            "image_id": image_ids[i],
            "image_path": image_paths[i],
            "true_label": idx_to_label[int(targets[i])],
            "pred_label": idx_to_label[int(preds[i])],
            "confidence": float(confidences[i]),
            "correct": bool(preds[i] == targets[i]),
            "needs_review": bool(confidences[i] < confidence_threshold),
        }

        for rank in range(top_k):
            class_idx = int(top_indices[i, rank])
            row[f"top_{rank + 1}_label"] = idx_to_label[class_idx]
            row[f"top_{rank + 1}_prob"] = float(probs[i, class_idx])

        # Full probability vector for calibration, ensembling, stacking,
        # and deeper error analysis.
        for class_idx in class_indices:
            class_name = idx_to_label[class_idx]
            row[f"prob_{class_name}"] = float(probs[i, class_idx])

        prediction_rows.append(row)

    predictions_path = report_dir / "predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_acc),
        "num_samples": int(len(targets)),
        "num_classes": int(len(class_names)),
        "mean_confidence": float(np.mean(confidences)),
        "needs_review_rate": float(np.mean(confidences < confidence_threshold)),
        "confidence_threshold": float(confidence_threshold),
        "per_class_metrics_csv": str(per_class_path),
        "confusion_matrix_csv": str(cm_path),
        "predictions_csv": str(predictions_path),
    }
    metrics.update(model_metadata)
    metrics.update(eval_metadata)

    metrics_path = report_dir / "test_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def make_run_metadata(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    split: str,
) -> tuple[str, dict[str, Any]]:
    checkpoint_config = checkpoint.get("config", config)

    backbone = str(checkpoint_config.get("train", {}).get("backbone", "unknown_model"))
    epoch = int(checkpoint.get("epoch", -1))
    best_val_macro_f1 = checkpoint.get("best_val_macro_f1", None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if epoch >= 0:
        run_name = f"{backbone}_{split}_epoch{epoch:03d}_{timestamp}"
    else:
        run_name = f"{backbone}_{split}_{timestamp}"

    model_metadata = {
        "run_name": run_name,
        "backbone": backbone,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "evaluation_split": str(config["data"][f"{split}_csv"]),
        "split": split,
    }

    return run_name, model_metadata


def make_eval_metadata(config: dict[str, Any]) -> dict[str, Any]:
    use_tta, tta_transforms = get_tta_config(config)

    return {
        "use_tta": use_tta,
        "tta_transforms": tta_transforms,
        "tta_n": len(tta_transforms),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = args.split

    artifact_dir = Path(config["project"].get("artifact_dir", "artifacts"))
    report_dir = artifact_dir / "reports"

    checkpoint_path = Path(config["inference"]["checkpoint_path"])
    device = get_device(config)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)

    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {int(idx): str(label) for label, idx in label_to_idx.items()}

    run_name, model_metadata = make_run_metadata(
        config=config,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        split = split
    )

    eval_metadata = make_eval_metadata(config)

    run_report_dir = report_dir / "runs" / run_name

    model_config = copy.deepcopy(checkpoint.get("config", config))
    model_config["train"]["pretrained"] = False

    model = build_model_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_loader = build_eval_dataloader(
        config=config,
        label_to_idx=label_to_idx,
        split=split,
    )

    probs, targets, image_ids, image_paths = predict(
        model=model,
        loader=eval_loader,
        device=device,
        config=config,
    )

    top_k = int(config["evaluate"].get("top_k", 3))
    confidence_threshold = float(config["evaluate"].get("confidence_threshold", 0.65))

    # 1. Latest artifacts
    metrics = save_eval_artifacts(
        report_dir=report_dir,
        probs=probs,
        targets=targets,
        image_ids=image_ids,
        image_paths=image_paths,
        idx_to_label=idx_to_label,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
        model_metadata=model_metadata,
        eval_metadata=eval_metadata,
    )

    # 2. Versioned artifacts
    save_eval_artifacts(
        report_dir=run_report_dir,
        probs=probs,
        targets=targets,
        image_ids=image_ids,
        image_paths=image_paths,
        idx_to_label=idx_to_label,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
        model_metadata=model_metadata,
        eval_metadata=eval_metadata,
    )

    print("[INFO] Evaluation finished.")
    print(f"[INFO] Latest reports saved to: {report_dir}")
    print(f"[INFO] Versioned reports saved to: {run_report_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()