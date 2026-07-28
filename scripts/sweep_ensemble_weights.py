from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

DEFAULT_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    lower_to_original = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in cols:
            return candidate
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def prob_columns(labels: list[str]) -> list[str]:
    return [f"prob_{label}" for label in labels]


def load_predictions(path: Path, labels: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = normalize_columns(pd.read_csv(path))

    image_col = find_col(df, ["image_id", "id", "image", "filename"])
    true_col = find_col(df, ["true_label", "label", "y_true", "target"])

    if image_col is None:
        raise ValueError(
            f"{path} does not contain an image_id/id column. Columns: {list(df.columns)}"
        )
    if true_col is None:
        raise ValueError(
            f"{path} does not contain a true_label/label column. Columns: {list(df.columns)}"
        )

    rename_map: dict[str, str] = {}
    if image_col != "image_id":
        rename_map[image_col] = "image_id"
    if true_col != "true_label":
        rename_map[true_col] = "true_label"
    if rename_map:
        df = df.rename(columns=rename_map)

    alt_map: dict[str, str] = {}
    for label in labels:
        canonical = f"prob_{label}"
        alt = f"{label}_prob"
        if canonical not in df.columns and alt in df.columns:
            alt_map[alt] = canonical
    if alt_map:
        df = df.rename(columns=alt_map)

    expected_prob_cols = prob_columns(labels)
    missing = [c for c in expected_prob_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing probability columns: {missing}. "
            f"Expected: {expected_prob_cols}. Columns: {list(df.columns)}"
        )

    out = df[["image_id", "true_label"] + expected_prob_cols].copy()
    out["image_id"] = out["image_id"].astype(str)
    out["true_label"] = out["true_label"].astype(str)

    probs = out[expected_prob_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all():
        raise ValueError(f"{path} contains non-finite probabilities.")
    if (probs < -1e-8).any():
        raise ValueError(f"{path} contains negative probabilities.")

    denom = np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    out.loc[:, expected_prob_cols] = probs / denom

    return out


def align_prediction_frames(
    model_frames: dict[str, pd.DataFrame],
    labels: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Align predictions by image_id and validate true labels match."""
    if not model_frames:
        raise ValueError("No model prediction frames provided.")

    model_names = list(model_frames.keys())
    base_name = model_names[0]

    base = model_frames[base_name][["image_id", "true_label"]].copy()
    base = base.sort_values("image_id").reset_index(drop=True)

    probs_by_model: dict[str, np.ndarray] = {}
    prob_cols = prob_columns(labels)

    for name, df in model_frames.items():
        cur = df[["image_id", "true_label"] + prob_cols].copy()
        cur = cur.sort_values("image_id").reset_index(drop=True)

        merged = base.merge(cur, on="image_id", how="inner", suffixes=("_base", ""))
        if len(merged) != len(base):
            missing_n = len(base) - len(merged)
            raise ValueError(
                f"Model {name} does not align with {base_name}: {missing_n} image_id(s) missing."
            )

        if not (merged["true_label_base"].astype(str) == merged["true_label"].astype(str)).all():
            bad = merged[
                merged["true_label_base"].astype(str) != merged["true_label"].astype(str)
            ].head()
            raise ValueError(f"True labels mismatch for model {name}:\n{bad}")

        probs = merged[prob_cols].to_numpy(dtype=np.float64)
        denom = np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        probs_by_model[name] = probs / denom

    return base, probs_by_model


def load_split_predictions(
    models_cfg: list[dict[str, Any]],
    split: str,
    labels: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    key = f"{split}_predictions"
    frames: dict[str, pd.DataFrame] = {}

    for model_cfg in models_cfg:
        name = model_cfg["name"]
        if key not in model_cfg:
            raise KeyError(f"Model config {name} is missing key: {key}")
        path = Path(model_cfg[key])
        frames[name] = load_predictions(path, labels)

    return align_prediction_frames(frames, labels)


# -----------------------------------------------------------------------------
# Ensemble weight generation
# -----------------------------------------------------------------------------


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {k: float(v) for k, v in weights.items() if float(v) > 0}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError(f"Invalid weights: {weights}")
    return {k: v / total for k, v in cleaned.items()}


def quantized_weight_values(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError(f"step must be in (0, 1], got {step}")
    n = int(round(1.0 / step))
    if abs(n * step - 1.0) > 1e-8:
        raise ValueError(f"step must divide 1.0 cleanly, got {step}")
    return [round(i * step, 10) for i in range(n + 1)]


def generate_weights_for_subset(
    subset: list[str],
    *,
    step: float,
    min_nonzero_weight: float,
    max_weight_by_model: dict[str, float],
    min_weight_by_model_when_present: dict[str, float],
) -> list[dict[str, float]]:
    """Generate weights for a fixed model subset.

    Every model in subset is used with weight >= min_nonzero_weight, weights sum to 1.
    """
    values = quantized_weight_values(step)
    results: list[dict[str, float]] = []

    for raw in product(values, repeat=len(subset)):
        if abs(sum(raw) - 1.0) > 1e-8:
            continue

        weights = {
            model: float(w)
            for model, w in zip(
                subset,
                raw,
                strict=True,
            )
        }
        if any(w < min_nonzero_weight for w in weights.values()):
            continue

        valid = True
        for model, w in weights.items():
            if w > float(max_weight_by_model.get(model, 1.0)) + 1e-12:
                valid = False
                break
            if w < float(min_weight_by_model_when_present.get(model, 0.0)) - 1e-12:
                valid = False
                break

        if not valid:
            continue

        results.append({k: round(v, 4) for k, v in weights.items()})

    return results


def short_model_name(model: str) -> str:
    return {
        "convnextv2_gmu": "gmu",
        "convnextv2_image_only": "img",
        "convnextv2_concat": "concat",
        "efficientnetv2_stage2": "eff",
        "dinov2_frozen": "dino",
    }.get(model, model)


def weight_set_name(weights: dict[str, float]) -> str:
    parts = []
    for model, weight in weights.items():
        pct = int(round(weight * 100))
        parts.append(f"{short_model_name(model)}{pct}")
    return "auto_" + "_".join(parts)


def generate_auto_weight_sets(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    auto_cfg = cfg.get("auto_weight_search", {}) or {}
    if not auto_cfg.get("enabled", False):
        return []

    step = float(auto_cfg.get("step", 0.05))
    min_nonzero_weight = float(auto_cfg.get("min_nonzero_weight", step))
    max_models_per_ensemble = int(auto_cfg.get("max_models_per_ensemble", 3))
    max_weight_by_model = auto_cfg.get("max_weight_by_model", {}) or {}
    min_weight_by_model_when_present = auto_cfg.get("min_weight_by_model_when_present", {}) or {}
    subsets = auto_cfg.get("subsets", []) or []

    generated: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()

    for subset in subsets:
        subset = list(subset)
        if len(subset) == 0:
            continue
        if len(subset) > max_models_per_ensemble:
            continue

        for weights in generate_weights_for_subset(
            subset,
            step=step,
            min_nonzero_weight=min_nonzero_weight,
            max_weight_by_model=max_weight_by_model,
            min_weight_by_model_when_present=min_weight_by_model_when_present,
        ):
            key = tuple(sorted(weights.items()))
            if key in seen:
                continue
            seen.add(key)
            generated.append(
                {"name": weight_set_name(weights), "weights": weights, "source": "auto"}
            )

    return generated


# -----------------------------------------------------------------------------
# Ensemble / metrics
# -----------------------------------------------------------------------------


def ensemble_probs(probs_by_model: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    weights = normalize_weights(weights)

    out = None
    for model_name, weight in weights.items():
        if model_name not in probs_by_model:
            raise KeyError(
                f"Weight set references unknown model: {model_name}. "
                f"Available: {list(probs_by_model)}"
            )
        weighted = weight * probs_by_model[model_name]
        out = weighted if out is None else out + weighted

    if out is None:
        raise ValueError("Empty ensemble.")

    denom = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out / denom


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    probs: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mean_confidence": float(np.max(probs, axis=1).mean()),
    }

    for label in labels:
        label_report = report.get(label, {})
        metrics[f"{label}_precision"] = float(label_report.get("precision", 0.0))
        metrics[f"{label}_recall"] = float(label_report.get("recall", 0.0))
        metrics[f"{label}_f1"] = float(label_report.get("f1-score", 0.0))
        metrics[f"{label}_support"] = int(label_report.get("support", 0))

    return metrics


def predictions_dataframe(base: pd.DataFrame, probs: np.ndarray, labels: list[str]) -> pd.DataFrame:
    pred_idx = probs.argmax(axis=1)
    pred_labels = [labels[i] for i in pred_idx]
    confidence = probs.max(axis=1)

    out = base.copy()
    out["pred_label"] = pred_labels
    out["confidence"] = confidence

    for i, label in enumerate(labels):
        out[f"prob_{label}"] = probs[:, i]

    return out


def evaluate_weight_sets_on_split(
    *,
    base: pd.DataFrame,
    probs_by_model: dict[str, np.ndarray],
    weight_sets: list[dict[str, Any]],
    labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_true = base["true_label"].astype(str).tolist()

    for weight_set in weight_sets:
        name = weight_set["name"]
        source = weight_set.get("source", "manual")
        weights = normalize_weights(weight_set["weights"])

        probs = ensemble_probs(probs_by_model, weights)
        pred_idx = probs.argmax(axis=1)
        y_pred = [labels[i] for i in pred_idx]
        metrics = compute_metrics(y_true, y_pred, probs, labels)

        rows.append(
            {
                "ensemble_name": name,
                "source": source,
                "num_models": len(weights),
                "weights_json": json.dumps(weights, sort_keys=True),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def sort_sweep(df: pd.DataFrame, primary: str, tie_breakers: list[str]) -> pd.DataFrame:
    sort_cols = [primary] + [c for c in tie_breakers if c in df.columns]
    return df.sort_values(
        by=sort_cols,
        ascending=[False] * len(sort_cols),
        na_position="last",
    ).reset_index(drop=True)


def select_best_row(sweep_df: pd.DataFrame, primary: str, tie_breakers: list[str]) -> pd.Series:
    return sort_sweep(sweep_df, primary, tie_breakers).iloc[0]


def save_selected_outputs(
    *,
    output_dir: Path,
    output_cfg: dict[str, Any],
    best_name: str,
    best_weights: dict[str, float],
    val_base: pd.DataFrame,
    val_probs_by_model: dict[str, np.ndarray],
    test_base: pd.DataFrame,
    test_probs_by_model: dict[str, np.ndarray],
    labels: list[str],
) -> None:
    best_val_probs = ensemble_probs(val_probs_by_model, best_weights)
    best_val_df = predictions_dataframe(val_base, best_val_probs, labels)
    val_pred_path = output_dir / output_cfg.get(
        "val_predictions_csv", "ensemble_val_predictions.csv"
    )
    best_val_df.to_csv(val_pred_path, index=False)

    val_metrics = compute_metrics(
        best_val_df["true_label"].astype(str).tolist(),
        best_val_df["pred_label"].astype(str).tolist(),
        best_val_probs,
        labels,
    )
    write_json(
        output_dir / output_cfg.get("best_val_metrics_json", "ensemble_best_val_metrics.json"),
        {"ensemble_name": best_name, "weights": best_weights, **val_metrics},
    )

    test_probs = ensemble_probs(test_probs_by_model, best_weights)
    test_df = predictions_dataframe(test_base, test_probs, labels)
    test_pred_path = output_dir / output_cfg.get(
        "test_predictions_csv", "ensemble_test_predictions.csv"
    )
    test_df.to_csv(test_pred_path, index=False)

    test_metrics = compute_metrics(
        test_df["true_label"].astype(str).tolist(),
        test_df["pred_label"].astype(str).tolist(),
        test_probs,
        labels,
    )
    write_json(
        output_dir / output_cfg.get("test_metrics_json", "ensemble_test_metrics.json"),
        {"ensemble_name": best_name, "weights": best_weights, **test_metrics},
    )

    print(f"[DONE] Wrote selected val predictions: {val_pred_path}")
    print(f"[DONE] Wrote selected test predictions: {test_pred_path}")
    print(
        "[DONE] Selected test metrics: "
        f"macro_f1={test_metrics['macro_f1']:.4f}, "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}, "
        f"accuracy={test_metrics['accuracy']:.4f}"
    )



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep soft-voting ensemble weights on validation predictions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ensemble/phase2_ensemble.yaml"),
        help="Path to ensemble YAML config.",
    )
    parser.add_argument(
        "--diagnose-test-sweep",
        action="store_true",
        help=(
            "Also evaluate every weight set on test for diagnosis only. "
            "Do not use test sweep to select final weights."
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    labels = cfg.get("data", {}).get("labels", DEFAULT_LABELS)
    models_cfg = cfg["models"]

    manual_weight_sets = cfg.get("weight_sets", []) or []
    for ws in manual_weight_sets:
        ws.setdefault("source", "manual")

    auto_weight_sets = generate_auto_weight_sets(cfg)
    weight_sets = manual_weight_sets + auto_weight_sets

    if not weight_sets:
        raise ValueError("No weight sets found. Provide weight_sets or enable auto_weight_search.")

    selection_cfg = cfg.get("selection", {})
    primary_metric = selection_cfg.get("primary_metric", "macro_f1")
    tie_breakers = selection_cfg.get(
        "tie_breakers",
        ["balanced_accuracy", "mel_recall", "akiec_recall"],
    )

    output_cfg = cfg.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", "artifacts/reports/ensemble"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Manual weight sets: {len(manual_weight_sets)}")
    print(f"[INFO] Auto weight sets: {len(auto_weight_sets)}")
    print(f"[INFO] Total weight sets: {len(weight_sets)}")

    print("[INFO] Loading validation predictions...")
    val_base, val_probs_by_model = load_split_predictions(models_cfg, "val", labels)

    print("[INFO] Loading test predictions...")
    test_base, test_probs_by_model = load_split_predictions(models_cfg, "test", labels)

    val_sweep_df = evaluate_weight_sets_on_split(
        base=val_base,
        probs_by_model=val_probs_by_model,
        weight_sets=weight_sets,
        labels=labels,
    )
    val_sweep_df = sort_sweep(val_sweep_df, primary_metric, tie_breakers)

    sweep_csv = output_dir / output_cfg.get("sweep_csv", "ensemble_weight_sweep.csv")
    val_sweep_df.to_csv(sweep_csv, index=False)

    best = select_best_row(val_sweep_df, primary_metric, tie_breakers)
    best_name = str(best["ensemble_name"])
    best_weights = json.loads(str(best["weights_json"]))

    print(f"[DONE] Best validation ensemble: {best_name}")
    print(f"[DONE] Source: {best.get('source', 'manual')}")
    print(f"[DONE] Weights: {best_weights}")
    print(f"[DONE] {primary_metric}: {best[primary_metric]:.4f}")
    print(f"[DONE] Wrote validation sweep: {sweep_csv}")

    save_selected_outputs(
        output_dir=output_dir,
        output_cfg=output_cfg,
        best_name=best_name,
        best_weights=best_weights,
        val_base=val_base,
        val_probs_by_model=val_probs_by_model,
        test_base=test_base,
        test_probs_by_model=test_probs_by_model,
        labels=labels,
    )

    if args.diagnose_test_sweep:
        test_sweep_df = evaluate_weight_sets_on_split(
            base=test_base,
            probs_by_model=test_probs_by_model,
            weight_sets=weight_sets,
            labels=labels,
        )
        test_sweep_df = sort_sweep(test_sweep_df, primary_metric, tie_breakers)
        test_sweep_path = output_dir / output_cfg.get(
            "test_diagnostic_sweep_csv",
            "ensemble_weight_sweep_test_diagnostic.csv",
        )
        test_sweep_df.to_csv(test_sweep_path, index=False)

        cols = [
            "ensemble_name",
            "source",
            "num_models",
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
            "mel_recall",
            "akiec_recall",
            "weights_json",
        ]
        cols = [c for c in cols if c in test_sweep_df.columns]
        print("\n[DIAG] Top test weight sets:")
        print(test_sweep_df[cols].head(30).to_string(index=False))
        print(f"[DONE] Wrote test diagnostic sweep: {test_sweep_path}")


if __name__ == "__main__":
    main()
