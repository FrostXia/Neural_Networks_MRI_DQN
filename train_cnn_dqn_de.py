#!/usr/bin/env python3
"""
PyTorch implementation of a CNN/baseline-backbone + DQN active learning framework for Alzheimer's MRI classification.

Main components:
  - Manifest-based OASIS1 loader: reads image_path, label, subject_key
  - 3D MRI -> 2.5D multi-slice image: default is 10 sagittal + 10 coronal + 10 axial slices = 30 channels
  - CNN classifier with focal loss; supports current CNN, ResNet18, DenseNet121, and BasicCNN baselines
  - DQN active learning policy with scope-style z-scored advantage regularization
  - Optional grouped Differential Evolution (DE) random-key hyperparameter optimization
  - Subject-level train/val/test and k-fold CV
  - Per-split metrics, predictions, confusion matrices, ROC curves, active-learning plots, and optional full-test-set input-gradient / Grad-CAM heatmaps

This is designed to replace the TensorFlow/Keras demo implementation with PyTorch.
It is a faithful practical implementation of the paper's pipeline structure, but it is
not a guaranteed byte-for-byte reproduction of the authors' private experiment code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import tempfile
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision import models
except Exception:  # torchvision is only required for ResNet/DenseNet baselines
    models = None
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def gmean_from_cm(cm: np.ndarray) -> float:
    if cm.shape != (2, 2):
        return float("nan")
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    return float(np.sqrt(max(tpr, 0.0) * max(tnr, 0.0)))


def compute_binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": float(auc),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn + 1e-12)),
        "specificity": float(tn / (tn + fp + 1e-12)),
        "gmean": gmean_from_cm(cm),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_optimal_threshold(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    """Choose a validation threshold without collapsing to one predicted class.

    Earlier versions used ROC/Youden with a hard clip to [0.01, 0.99].
    For poorly calibrated MRI models this could accidentally choose 0.01,
    making every test case become class 1. This version searches candidate
    thresholds directly on the validation set, rejects thresholds that predict
    only one class whenever possible, and falls back to 0.5 if no safe
    threshold exists.
    """
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    mask = np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    try:
        scores = np.unique(np.clip(y_prob, 0.0, 1.0))
        candidates = [0.5]
        candidates.extend(np.linspace(0.05, 0.95, 19).tolist())
        if len(scores) == 1:
            candidates.append(float(scores[0]))
        else:
            mids = (scores[:-1] + scores[1:]) / 2.0
            candidates.extend(mids.tolist())
            candidates.extend(scores.tolist())
        candidates = sorted({float(t) for t in candidates if np.isfinite(t) and 0.0 <= float(t) <= 1.0})

        best_thr = 0.5
        best_score = -float("inf")
        found_noncollapsed = False
        n = len(y_true)
        for thr in candidates:
            y_pred = (y_prob >= thr).astype(int)
            pos = int(y_pred.sum())
            neg = int(n - pos)
            # Avoid all-positive or all-negative validation predictions.
            # If the model cannot produce both classes at any threshold, fallback to 0.5.
            if pos == 0 or neg == 0:
                continue
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn + 1e-12)
            specificity = tn / (tn + fp + 1e-12)
            balanced_acc = 0.5 * (sensitivity + specificity)
            # Tie-breaker: prefer thresholds near 0.5 to avoid extreme operating points.
            score = float(balanced_acc - 1e-3 * abs(float(thr) - 0.5))
            if score > best_score:
                best_score = score
                best_thr = float(thr)
                found_noncollapsed = True
        if not found_noncollapsed:
            return 0.5
        return float(best_thr)
    except Exception:
        return 0.5


def is_supervised_label_value(label) -> bool:
    try:
        if pd.isna(label):
            return False
        return int(float(label)) in {0, 1}
    except Exception:
        return False


def parse_bool_value(value) -> bool:
    """Parse bool-like CSV values safely.

    Important: pandas/object ``astype(bool)`` treats the string ``"False"`` as
    True. That can accidentally empty the external unlabeled pool, so we parse
    common string values explicitly.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y", "labeled"}:
        return True
    if text in {"false", "f", "0", "no", "n", "", "nan", "none", "null", "unlabeled"}:
        return False
    return bool(value)


def bool_mask_from_series(series: pd.Series) -> np.ndarray:
    return series.apply(parse_bool_value).to_numpy(dtype=bool)


def labeled_mask_from_df(df: pd.DataFrame) -> np.ndarray:
    if "is_labeled" in df.columns:
        return bool_mask_from_series(df["is_labeled"])
    return df["label"].apply(is_supervised_label_value).to_numpy(dtype=bool)


def reveal_label_or_zero(label) -> tuple[int, bool]:
    """Return the oracle label used after querying the AL pool.

    Existing labels 0/1 are preserved. Empty/NaN/-1 labels are treated as 0
    when they are queried/revealed, per the requested OASIS1 behavior. The
    boolean tells whether this was imputed from an empty unlabeled value.
    """
    if is_supervised_label_value(label):
        return int(float(label)), False
    return 0, True


def checkpoint_score(metrics: dict[str, float], metric: str = "composite") -> float:
    metric = str(metric or "composite").lower()
    auc = metrics.get("auc", float("nan"))
    if not np.isfinite(auc):
        auc = metrics.get("balanced_accuracy", 0.0)
    if metric == "composite":
        return float(
            0.35 * metrics.get("f1", 0.0)
            + 0.30 * auc
            + 0.20 * metrics.get("gmean", 0.0)
            + 0.15 * metrics.get("balanced_accuracy", 0.0)
        )
    value = metrics.get(metric, float("nan"))
    if not np.isfinite(value):
        return -float("inf")
    return float(value)


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def load_state_dict_cpu_safe(module: nn.Module, state: Optional[dict[str, torch.Tensor]], device: torch.device) -> None:
    if state is None:
        return
    module.load_state_dict({k: v.to(device) for k, v in state.items()})


# -----------------------------
# Output tables and figures
# -----------------------------


def plot_confusion_matrix(metrics: dict, out_path: Path) -> None:
    cm = np.array([[metrics.get("tn", 0), metrics.get("fp", 0)], [metrics.get("fn", 0), metrics.get("tp", 0)]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0\nnon-demented", "Pred 1\ndemented"])
    ax.set_yticklabels(["True 0\nnon-demented", "True 1\ndemented"])
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true: Iterable[int], y_prob: Iterable[float], out_path: Path) -> None:
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_val = float("nan")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / sensitivity")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cycle_metrics(history: list[dict[str, Any]], figures_dir: Path) -> None:
    if not history:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    plots = [
        ("al_auc_curve.png", ["train_auc", "val_auc"], "Active learning AUC by cycle", "auc"),
        ("al_f1_curve.png", ["train_f1", "val_f1"], "Active learning F1 by cycle", "f1"),
        ("al_accuracy_curve.png", ["train_accuracy", "val_accuracy"], "Active learning accuracy by cycle", "accuracy"),
        ("al_sensitivity_specificity_curve.png", ["val_sensitivity", "val_specificity"], "Validation sensitivity/specificity by cycle", "score"),
        ("labeled_count_curve.png", ["labeled_count"], "Labeled samples by cycle", "count"),
        ("dqn_loss_curve.png", ["dqn_dqn_loss", "dqn_td_loss", "dqn_scope_loss"], "DQN losses by cycle", "loss"),
        ("epsilon_curve.png", ["dqn_epsilon"], "DQN epsilon by cycle", "epsilon"),
    ]
    for filename, cols, title, ylabel in plots:
        existing = [c for c in cols if c in hist.columns]
        if not existing or "cycle" not in hist.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for col in existing:
            ax.plot(hist["cycle"], hist[col], marker="o", label=col)
        ax.set_xlabel("Active-learning cycle")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=200)
        plt.close(fig)


def plot_epoch_metrics(epoch_history: list[dict[str, Any]], figures_dir: Path) -> None:
    """Plot per-epoch train/val curves inside each active-learning stage."""
    if not epoch_history:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(epoch_history)
    if "global_epoch" not in hist.columns:
        return
    plots = [
        ("epoch_loss_curve.png", ["train_loss", "val_loss"], "Classifier loss by epoch", "loss"),
        ("epoch_auc_curve.png", ["train_auc", "val_auc"], "Classifier AUC by epoch", "auc"),
        ("epoch_f1_curve.png", ["train_f1", "val_f1"], "Classifier F1 by epoch", "f1"),
        ("epoch_accuracy_curve.png", ["train_accuracy", "val_accuracy"], "Classifier accuracy by epoch", "accuracy"),
        ("epoch_sensitivity_specificity_curve.png", ["val_sensitivity", "val_specificity"], "Validation sensitivity/specificity by epoch", "score"),
    ]
    for filename, cols, title, ylabel in plots:
        existing = [c for c in cols if c in hist.columns]
        if not existing:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for col in existing:
            ax.plot(hist["global_epoch"], hist[col], label=col)
        # Mark active-learning retraining boundaries.
        if "cycle" in hist.columns:
            boundaries = hist.groupby("cycle")["global_epoch"].min().sort_index().tolist()
            for b in boundaries[1:]:
                ax.axvline(b, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Global classifier epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=200)
        plt.close(fig)


def plot_selected_samples(selected_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    if not selected_rows:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    df_sel = pd.DataFrame(selected_rows)
    if "cycle" in df_sel.columns and "entropy" in df_sel.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_sel.boxplot(column="entropy", by="cycle", ax=ax)
        ax.set_title("Selected-sample entropy by cycle")
        ax.set_xlabel("Active-learning cycle")
        ax.set_ylabel("entropy")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(figures_dir / "selected_entropy_by_cycle.png", dpi=200)
        plt.close(fig)
    if "label" in df_sel.columns and "cycle" in df_sel.columns:
        counts = df_sel.groupby(["cycle", "label"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Selected labels by cycle")
        ax.set_xlabel("Active-learning cycle")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(figures_dir / "selected_label_counts_by_cycle.png", dpi=200)
        plt.close(fig)


def plot_split_label_counts(split_summary: dict[str, dict[str, Any]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, info in split_summary.items():
        for label, count in info.get("label_counts_images", {}).items():
            rows.append({"split": split, "label": str(label), "count": int(count)})
    if not rows:
        return
    counts = pd.DataFrame(rows).pivot_table(index="split", columns="label", values="count", fill_value=0, aggfunc="sum")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Label counts by split")
    ax.set_xlabel("split")
    ax.set_ylabel("image count")
    fig.tight_layout()
    fig.savefig(figures_dir / "split_label_counts.png", dpi=200)
    plt.close(fig)


def save_split_outputs(
    model: "CNNClassifier2D",
    X_all: np.ndarray,
    y_all: np.ndarray,
    indices: np.ndarray,
    df: pd.DataFrame,
    split_name: str,
    fold_dir: Path,
    args,
    device: torch.device,
    threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    metrics, pred_df = evaluate_classifier(model, X_all[indices], y_all[indices], args.batch_size, device, threshold=threshold)
    pred_df = pred_df.reset_index(drop=True)
    meta = df.iloc[indices].reset_index(drop=True)
    out = pd.concat([meta, pred_df], axis=1)
    out["prediction_name"] = ["demented" if int(p) == 1 else "non_demented" for p in out["y_pred"]]
    out["eval_split"] = split_name

    eval_dir = fold_dir / f"{split_name}_eval"
    metrics_dir = eval_dir / "metrics"
    figures_dir = eval_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, metrics_dir / f"{split_name}_metrics.json")
    # Compatibility with the previous evaluate.py naming, even when split_name is train/val.
    save_json(metrics, metrics_dir / "test_metrics.json")
    out.to_csv(metrics_dir / f"{split_name}_predictions.csv", index=False)
    out.to_csv(metrics_dir / "test_predictions.csv", index=False)

    cm_df = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["true_non_demented", "true_demented"],
        columns=["pred_non_demented", "pred_demented"],
    )
    cm_df.to_csv(metrics_dir / "confusion_matrix.csv")
    plot_confusion_matrix(metrics, figures_dir / "confusion_matrix.png")
    plot_roc_curve(out["y_true"].to_numpy(), out["prob_demented"].to_numpy(), figures_dir / "roc_curve.png")

    # Convenience copies at fold root.
    save_json(metrics, fold_dir / f"{split_name}_metrics.json")
    out.to_csv(fold_dir / f"{split_name}_predictions.csv", index=False)
    return metrics, out


# -----------------------------
# Full test-set saliency / Grad-CAM heatmaps for the 2.5D multi-slice classifier
# -----------------------------


def _last_conv2d_module(model: nn.Module) -> nn.Module:
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last


class GradCAM2D:
    def __init__(self, model: nn.Module, target_module: Optional[nn.Module] = None):
        self.model = model
        self.target_module = target_module or _last_conv2d_module(model)
        self.activations: Optional[torch.Tensor] = None
        # Use tensor.retain_grad() on the hooked activation instead of a full
        # backward hook. It is less fragile across PyTorch versions and avoids
        # applying one global CAM to multiple anatomical planes.
        self._fwd = self.target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if torch.is_tensor(output) and output.requires_grad:
            output.retain_grad()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        logits = self.model(x)
        score = logits[:, int(class_idx)].sum()
        score.backward(retain_graph=False)
        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("Grad-CAM hook did not capture activation gradients.")
        gradients = self.activations.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        return _normalize_heatmap(cam)

    def close(self) -> None:
        self._fwd.remove()


def _safe_filename(text: str, fallback: str) -> str:
    text = str(text).strip() or fallback
    text = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in text)
    return text[:160] or fallback


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = np.asarray(hm, dtype=np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = hm - float(hm.min())
    mx = float(hm.max())
    if mx > 1e-8:
        hm = hm / mx
    return hm.astype(np.float32)


def _normalize_heatmap_chw(hm_chw: np.ndarray) -> np.ndarray:
    hm_chw = np.asarray(hm_chw, dtype=np.float32)
    out = np.zeros_like(hm_chw, dtype=np.float32)
    for c in range(hm_chw.shape[0]):
        out[c] = _normalize_heatmap(hm_chw[c])
    return out


def compute_input_gradient_saliency(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Return one saliency heatmap per input channel/slice.

    For a 30-channel 2.5D input, the returned array has shape (30, H, W):
    every sagittal/coronal/axial slice gets its own heatmap. The model still
    receives the full multi-slice tensor; saliency is computed from the target
    class score gradient with respect to each input channel.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    x_req = x.detach().clone().requires_grad_(True)
    logits = model(x_req)
    score = logits[:, int(class_idx)].sum()
    score.backward(retain_graph=False)
    grad = x_req.grad.detach()
    # Gradient * input tends to be less noisy than raw |gradient| for image saliency.
    sal = (grad * x_req).abs()[0].detach().cpu().numpy().astype(np.float32)
    if sal.ndim == 4:  # 2D mode: (S, 3, H, W) -> one heatmap per slice
        sal = sal.sum(axis=1)
    return _normalize_heatmap_chw(sal)


def compute_branch_gradcams(model: nn.Module, x: torch.Tensor, class_idx: int) -> list[dict[str, Any]]:
    """Compute one Grad-CAM per CNN feature map.

    For non-branch backbones (current, BasicCNN, ResNet18, DenseNet121), this
    returns a single global feature-level Grad-CAM. It is not slice-specific;
    use input-gradient for per-slice maps.
    """
    out: list[dict[str, Any]] = []
    if hasattr(model, "branches") and hasattr(model, "channel_slices"):
        for branch_idx, (branch, (start, end)) in enumerate(zip(model.branches, model.channel_slices)):
            helper = GradCAM2D(model, target_module=_last_conv2d_module(branch))
            try:
                cam = helper(x, class_idx=class_idx)
            finally:
                helper.close()
            out.append({"branch_idx": int(branch_idx), "start": int(start), "end": int(end), "cam": cam})
        return out

    # Current CNN: a single global feature-level Grad-CAM.
    helper = GradCAM2D(model)
    try:
        cam = helper(x, class_idx=class_idx)
    finally:
        helper.close()
    out.append({"branch_idx": 0, "start": 0, "end": int(x.shape[1]), "cam": cam})
    return out


def _display_channel_for_plane(args, plane_idx: int) -> tuple[int, float, int]:
    """Return (channel_index, slice_fraction, local_slice_number) to display for one anatomical plane."""
    n = max(1, int(getattr(args, "slices_per_plane", 1)))
    fracs = _slice_fractions_for_count(n, float(getattr(args, "slice_fraction_min", 0.15)), float(getattr(args, "slice_fraction_max", 0.85)))
    display_indices = getattr(args, "heatmap_display_indices", None)
    if display_indices is not None and all(int(v) >= 0 for v in display_indices):
        local = int(np.clip(int(display_indices[plane_idx]), 0, n - 1))
    else:
        display_fracs = getattr(args, "heatmap_display_fractions", (0.5, 0.5, 0.5))
        target = float(display_fracs[plane_idx])
        local = int(np.argmin(np.abs(np.asarray(fracs, dtype=np.float32) - target)))
    channel = plane_idx * n + local
    return int(channel), float(fracs[local]), int(local)


def _channel_metadata(args, channel: int) -> tuple[str, int, float]:
    n = max(1, int(getattr(args, "slices_per_plane", 1)))
    plane_names = ["sagittal", "coronal", "axial"]
    plane_idx = min(channel // n, 2) if n > 0 else 0
    local = int(channel - plane_idx * n)
    fracs = _slice_fractions_for_count(n, float(getattr(args, "slice_fraction_min", 0.15)), float(getattr(args, "slice_fraction_max", 0.85)))
    frac = float(fracs[int(np.clip(local, 0, n - 1))]) if fracs else float("nan")
    return plane_names[plane_idx] if plane_idx < len(plane_names) else f"plane{plane_idx}", local, frac


def save_three_slice_saliency_png(
    x_chw: np.ndarray,
    saliency_chw: np.ndarray,
    row: pd.Series,
    out_path: Path,
    args,
    class_idx: int,
) -> None:
    """Save a 3-panel summary using slice-specific input-gradient saliency.

    Unlike the legacy single-CAM overlay, each displayed slice uses its own
    channel-specific heatmap from saliency_chw[channel].
    """
    plane_names = ["sagittal", "coronal", "axial"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for plane_idx, ax in enumerate(axes):
        ch, frac, local_slice = _display_channel_for_plane(args, plane_idx)
        ch = min(ch, x_chw.shape[0] - 1)
        ax.imshow(x_chw[ch], cmap="gray")
        ax.imshow(saliency_chw[ch], cmap="jet", alpha=float(args.heatmap_alpha))
        ax.set_title(f"{plane_names[plane_idx]} saliency\nch={ch}, slice#{local_slice}, frac={frac:.2f}")
        ax.axis("off")
    title = (
        f"{row.get('mri_id', '')} | true={row.get('y_true', '')} pred={row.get('y_pred', '')} "
        f"prob_AD={float(row.get('prob_demented', float('nan'))):.4f} | input-gradient class={class_idx} | input_ch={x_chw.shape[0]}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.heatmap_dpi), bbox_inches="tight")
    plt.close(fig)


def _display_channel_for_branch(args, branch_info: dict[str, Any], x_channels: int, n_branches: int) -> tuple[int, str]:
    start = int(branch_info.get("start", 0))
    end = int(branch_info.get("end", x_channels))
    branch_idx = int(branch_info.get("branch_idx", 0))
    plane_names = ["sagittal", "coronal", "axial"]

    # Common paper-aligned case: 3 branches map to the 3 anatomical planes.
    if n_branches == 3 and branch_idx < 3:
        ch, frac, local_slice = _display_channel_for_plane(args, branch_idx)
        if start <= ch < end:
            return int(ch), f"{plane_names[branch_idx]} branch\nch={ch}, slice#{local_slice}, frac={frac:.2f}"

    # General fallback: display the middle channel inside this branch's channel group.
    if end <= start:
        ch = min(max(start, 0), x_channels - 1)
    else:
        ch = int(np.clip((start + end - 1) // 2, 0, x_channels - 1))
    plane, local_slice, frac = _channel_metadata(args, ch)
    return ch, f"branch {branch_idx}\n{plane} ch={ch}, slice#{local_slice}, frac={frac:.2f}"


def save_branch_gradcam_png(
    x_chw: np.ndarray,
    branch_cams: list[dict[str, Any]],
    row: pd.Series,
    out_path: Path,
    args,
    class_idx: int,
) -> None:
    """Save branch/plane-level Grad-CAM overlays.

    Each CNN feature map gets its own CAM. With --cnn-stacks 3 this typically
    means one sagittal-level CAM, one coronal-level CAM, and one axial-level CAM.
    It is not a per-slice heatmap.
    """
    n = max(1, len(branch_cams))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, info in zip(axes, branch_cams):
        ch, title = _display_channel_for_branch(args, info, x_chw.shape[0], n)
        ax.imshow(x_chw[ch], cmap="gray")
        ax.imshow(info["cam"], cmap="jet", alpha=float(args.heatmap_alpha))
        ax.set_title(title)
        ax.axis("off")
    title = (
        f"{row.get('mri_id', '')} | true={row.get('y_true', '')} pred={row.get('y_pred', '')} "
        f"prob_AD={float(row.get('prob_demented', float('nan'))):.4f} | branch Grad-CAM class={class_idx}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.heatmap_dpi), bbox_inches="tight")
    plt.close(fig)


def save_all_slice_saliency_pngs(
    x_chw: np.ndarray,
    saliency_chw: np.ndarray,
    out_dir: Path,
    args,
) -> None:
    """Optionally save one small PNG per input channel/slice."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ch in range(x_chw.shape[0]):
        plane, local_slice, frac = _channel_metadata(args, ch)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(x_chw[ch], cmap="gray")
        ax.imshow(saliency_chw[ch], cmap="jet", alpha=float(args.heatmap_alpha))
        ax.set_title(f"{plane} | ch={ch} | slice#{local_slice} | frac={frac:.2f}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"ch{ch:03d}_{plane}_slice{local_slice:02d}.png", dpi=int(args.heatmap_dpi), bbox_inches="tight")
        plt.close(fig)



def save_2d_slice_saliency_png(
    x_schw: np.ndarray,
    saliency_shw: np.ndarray,
    row: pd.Series,
    out_path: Path,
    args,
    class_idx: int,
) -> None:
    """Save first/middle/last 2D slice saliency summary for --input-mode 2d."""
    x_schw = np.asarray(x_schw, dtype=np.float32)
    saliency_shw = np.asarray(saliency_shw, dtype=np.float32)
    n = int(x_schw.shape[0])
    picks = sorted({0, n // 2, max(0, n - 1)})
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4))
    if len(picks) == 1:
        axes = [axes]
    for ax, si in zip(axes, picks):
        img = x_schw[si].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img, cmap="gray")
        ax.imshow(saliency_shw[si], cmap="jet", alpha=float(args.heatmap_alpha))
        ax.set_title(f"{getattr(args, 'plane', 'axial')} slice {si + 1}/{n}")
        ax.axis("off")
    title = (
        f"{row.get('mri_id', '')} | true={row.get('y_true', '')} pred={row.get('y_pred', '')} "
        f"prob_AD={float(row.get('prob_demented', float('nan'))):.4f} | 2D input-gradient class={class_idx}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.heatmap_dpi), bbox_inches="tight")
    plt.close(fig)


def save_2d_gradcam_png(
    x_chw: np.ndarray,
    cam: np.ndarray,
    row: pd.Series,
    out_path: Path,
    args,
    class_idx: int,
) -> None:
    img = np.asarray(x_chw, dtype=np.float32).transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("MRI slice")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(cam, cmap="jet", alpha=float(args.heatmap_alpha))
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    title = (
        f"{row.get('mri_id', '')} | true={row.get('y_true', '')} pred={row.get('y_pred', '')} "
        f"prob_AD={float(row.get('prob_demented', float('nan'))):.4f} | 2D Grad-CAM class={class_idx}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.heatmap_dpi), bbox_inches="tight")
    plt.close(fig)

def save_full_test_heatmaps(
    model: nn.Module,
    X_all: np.ndarray,
    y_all: np.ndarray,
    test_indices: np.ndarray,
    test_predictions: pd.DataFrame,
    fold_dir: Path,
    args,
    device: torch.device,
) -> None:
    """Save explanation overlays for every test-set image.

    Default method is input-gradient saliency: each input slice/channel gets its
    own heatmap. Optional branch-gradcam saves one feature-level Grad-CAM per
    CNN feature map. The legacy behavior of applying one global Grad-CAM
    to multiple anatomical planes is intentionally not used as the default.
    """
    heatmap_root = fold_dir / "test_eval" / "heatmaps"
    heatmap_root.mkdir(parents=True, exist_ok=True)
    method = str(getattr(args, "heatmap_method", "input-gradient")).lower().replace("_", "-")
    do_saliency = method in {"input-gradient", "both"}
    do_branch_gradcam = method in {"branch-gradcam", "both"}
    if method not in {"input-gradient", "branch-gradcam", "both"}:
        raise ValueError(f"Unsupported --heatmap-method: {method}")

    summary_rows: list[dict[str, Any]] = []
    test_predictions = test_predictions.reset_index(drop=True)
    desc = f"save test heatmaps ({method})"
    for local_i, global_idx in enumerate(tqdm(test_indices, desc=desc, leave=False)):
        pred_row = test_predictions.iloc[local_i].copy()
        if args.heatmap_target == "true":
            class_idx = int(y_all[int(global_idx)])
        elif args.heatmap_target == "demented":
            class_idx = 1
        else:
            class_idx = int(pred_row.get("y_pred", 0))
        x_np = X_all[int(global_idx)]
        x_t = torch.from_numpy(x_np[None, ...]).float().to(device)
        mri_id = _safe_filename(pred_row.get("mri_id", f"idx_{global_idx}"), f"idx_{global_idx}")
        case_type = "TP" if int(pred_row.get("y_true", 0)) == 1 and int(pred_row.get("y_pred", 0)) == 1 else \
                    "TN" if int(pred_row.get("y_true", 0)) == 0 and int(pred_row.get("y_pred", 0)) == 0 else \
                    "FP" if int(pred_row.get("y_true", 0)) == 0 and int(pred_row.get("y_pred", 0)) == 1 else "FN"

        row_out: dict[str, Any] = {
            "local_test_index": int(local_i),
            "manifest_index": int(global_idx),
            "mri_id": pred_row.get("mri_id", ""),
            "subject_key": pred_row.get("subject_key", ""),
            "dataset": pred_row.get("dataset", ""),
            "image_path": pred_row.get("image_path", ""),
            "y_true": int(pred_row.get("y_true", y_all[int(global_idx)])),
            "y_pred": int(pred_row.get("y_pred", 0)),
            "prob_demented": float(pred_row.get("prob_demented", float("nan"))),
            "case_type": case_type,
            "heatmap_target_class": int(class_idx),
            "heatmap_method": method,
            "input_mode": str(getattr(args, "input_mode", "2.5d")),
            "plane": str(getattr(args, "plane", "")),
            "n_slices": int(getattr(args, "n_slices", 0)),
            "slices_per_plane": int(args.slices_per_plane),
            "input_channels": int(X_all.shape[2] if x_np.ndim == 4 else X_all.shape[1]),
            "slice_fraction_min": float(args.slice_fraction_min),
            "slice_fraction_max": float(args.slice_fraction_max),
            "heatmap_display_fractions": ",".join(map(str, args.heatmap_display_fractions)),
            "heatmap_display_indices": ",".join(map(str, args.heatmap_display_indices)) if args.heatmap_display_indices is not None else "",
        }

        if do_saliency:
            saliency = compute_input_gradient_saliency(model, x_t, class_idx=class_idx)
            out_png = heatmap_root / "saliency_slice_level" / case_type / f"{local_i:04d}_{mri_id}_saliency_class{class_idx}.png"
            if x_np.ndim == 4:
                save_2d_slice_saliency_png(x_np, saliency, pred_row, out_png, args, class_idx=class_idx)
            else:
                save_three_slice_saliency_png(x_np, saliency, pred_row, out_png, args, class_idx=class_idx)
            row_out["saliency_png"] = str(out_png)
            if args.heatmap_save_npy:
                npy_path = out_png.with_suffix(".npy")
                np.save(npy_path, saliency)
                row_out["saliency_npy"] = str(npy_path)
            if getattr(args, "heatmap_save_all_slice_pngs", False) and x_np.ndim != 4:
                all_dir = out_png.with_suffix("")
                save_all_slice_saliency_pngs(x_np, saliency, all_dir, args)
                row_out["saliency_all_slice_dir"] = str(all_dir)

        if do_branch_gradcam:
            out_png = heatmap_root / "gradcam_branch_level" / case_type / f"{local_i:04d}_{mri_id}_branch_gradcam_class{class_idx}.png"
            if x_np.ndim == 4:
                mid = int(x_np.shape[0] // 2)
                base_model = model.base if hasattr(model, "base") else model
                x_mid = torch.from_numpy(x_np[mid][None, ...]).float().to(device)
                branch_cams = compute_branch_gradcams(base_model, x_mid, class_idx=class_idx)
                cam = branch_cams[0]["cam"] if branch_cams else np.zeros(x_np.shape[-2:], dtype=np.float32)
                save_2d_gradcam_png(x_np[mid], cam, pred_row, out_png, args, class_idx=class_idx)
            else:
                branch_cams = compute_branch_gradcams(model, x_t, class_idx=class_idx)
                save_branch_gradcam_png(x_np, branch_cams, pred_row, out_png, args, class_idx=class_idx)
            row_out["branch_gradcam_png"] = str(out_png)
            row_out["branch_gradcam_count"] = int(len(branch_cams))
            if args.heatmap_save_npy:
                npy_path = out_png.with_suffix(".npy")
                np.save(npy_path, np.stack([b["cam"] for b in branch_cams], axis=0).astype(np.float32))
                row_out["branch_gradcam_npy"] = str(npy_path)

        summary_rows.append(row_out)

    pd.DataFrame(summary_rows).to_csv(heatmap_root / "heatmap_index.csv", index=False)


# -----------------------------
# MRI manifest loading
# -----------------------------


def _resize_2d_numpy(img2d: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D array using torch interpolation to avoid extra dependencies."""
    arr = np.asarray(img2d, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.from_numpy(arr)[None, None, ...].float()
    t = F.interpolate(t, size=image_size, mode="bilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float32)


def _normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.squeeze(vol)
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume, got shape={vol.shape}")
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = vol[np.abs(vol) > 1e-6]
    if nonzero.size < 32:
        nonzero = vol.reshape(-1)
    lo, hi = np.percentile(nonzero, [1, 99]) if nonzero.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    return vol.astype(np.float32)


def _resolve_slice_index(dim: int, fraction: float, explicit_index: Optional[int] = None) -> int:
    """Convert a normalized slice fraction or explicit index into a valid 0-based slice index."""
    if dim <= 0:
        return 0
    if explicit_index is not None and explicit_index >= 0:
        return int(np.clip(explicit_index, 0, dim - 1))
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return int(np.clip(round(fraction * (dim - 1)), 0, dim - 1))


def _slice_fractions_for_count(n: int, lo: float = 0.15, hi: float = 0.85) -> list[float]:
    """Evenly spaced slice fractions, avoiding extreme edges by default."""
    n = max(1, int(n))
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    if hi < lo:
        lo, hi = hi, lo
    if n == 1:
        return [0.5]
    if abs(hi - lo) < 1e-6:
        hi = min(1.0, lo + 1e-3)
    return [float(x) for x in np.linspace(lo, hi, n)]


def _slice_2d_from_volume(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:      # sagittal
        return np.rot90(vol[index, :, :])
    if axis == 1:      # coronal
        return np.rot90(vol[:, index, :])
    return np.rot90(vol[:, :, index])  # axial


def load_mri_multi_slices(
    path: str | Path,
    image_size: tuple[int, int] = (128, 128),
    slices_per_plane: int = 10,
    slice_fraction_min: float = 0.15,
    slice_fraction_max: float = 0.85,
    slice_fractions: tuple[float, float, float] = (0.5, 0.5, 0.5),
    slice_indices: Optional[tuple[int, int, int]] = None,
) -> np.ndarray:
    """Load a 3D MRI path and return a 2.5D multi-channel image.

    Default output uses 30 channels:
        channels 0..9   = sagittal slices
        channels 10..19 = coronal slices
        channels 20..29 = axial slices

    Set --slices-per-plane 1 to reproduce the old 3-channel behavior.
    Output shape is (3 * slices_per_plane, H, W), normalized to [0, 1].
    Supports .nii, .nii.gz, .hdr/.img Analyze pairs, .mgz, and common 2D images.
    """
    n = max(1, int(slices_per_plane))
    total_channels = 3 * n
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        from PIL import Image

        img = Image.open(path).convert("L").resize((image_size[1], image_size[0]))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.repeat(arr[None, ...], total_channels, axis=0).astype(np.float32)

    try:
        import nibabel as nib
    except Exception as e:
        raise ImportError("nibabel is required for MRI volume loading. Install with: pip install nibabel") from e

    img = nib.load(str(path))
    vol = _normalize_volume(img.get_fdata(dtype=np.float32))

    planes: list[np.ndarray] = []
    if n == 1:
        # Backward-compatible 3-channel mode: one manually chosen slice per anatomical plane.
        if slice_indices is None:
            slice_indices = (-1, -1, -1)
        for axis in range(3):
            idx = _resolve_slice_index(vol.shape[axis], slice_fractions[axis], slice_indices[axis])
            planes.append(_resize_2d_numpy(_slice_2d_from_volume(vol, axis, idx), image_size))
    else:
        fracs = _slice_fractions_for_count(n, slice_fraction_min, slice_fraction_max)
        for axis in range(3):
            for frac in fracs:
                idx = _resolve_slice_index(vol.shape[axis], frac, None)
                planes.append(_resize_2d_numpy(_slice_2d_from_volume(vol, axis, idx), image_size))
    return np.stack(planes, axis=0).astype(np.float32)


def load_mri_2d_slices(
    path: str | Path,
    image_size: tuple[int, int] = (224, 224),
    plane: str = "axial",
    n_slices: int = 15,
    slice_fraction_min: float = 0.15,
    slice_fraction_max: float = 0.85,
) -> np.ndarray:
    """Load a 3D MRI path as a true 2D multi-slice input.

    Output shape is (n_slices, 3, H, W). Each grayscale MRI slice is
    repeated to 3 channels so the original ResNet18/DenseNet121/BasicCNN
    baselines can be used without changing their RGB stems. The classifier
    wrapper averages slice logits/features back to one MRI/patient-level
    prediction, so active learning still selects MRI rows rather than
    individual slices.
    """
    n = max(1, int(n_slices))
    plane_name = str(plane).lower()
    axis_map = {"sagittal": 0, "coronal": 1, "axial": 2}
    if plane_name not in axis_map:
        raise ValueError(f"Unsupported 2D plane: {plane}. Choose sagittal, coronal, or axial.")

    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        from PIL import Image
        img = Image.open(path).convert("L").resize((image_size[1], image_size[0]))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        rgb = np.repeat(arr[None, ...], 3, axis=0).astype(np.float32)
        return np.repeat(rgb[None, ...], n, axis=0).astype(np.float32)

    try:
        import nibabel as nib
    except Exception as e:
        raise ImportError("nibabel is required for MRI volume loading. Install with: pip install nibabel") from e

    img = nib.load(str(path))
    vol = _normalize_volume(img.get_fdata(dtype=np.float32))
    axis = axis_map[plane_name]
    fracs = _slice_fractions_for_count(n, slice_fraction_min, slice_fraction_max)
    slices: list[np.ndarray] = []
    for frac in fracs:
        idx = _resolve_slice_index(vol.shape[axis], frac, None)
        sl = _resize_2d_numpy(_slice_2d_from_volume(vol, axis, idx), image_size)
        slices.append(np.repeat(sl[None, ...], 3, axis=0).astype(np.float32))
    return np.stack(slices, axis=0).astype(np.float32)

def validate_manifest(df: pd.DataFrame) -> None:
    required = {"image_path", "label", "subject_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Manifest is empty.")
    # label -1 means externally unlabeled; these rows can enter the AL pool but
    # are never used in supervised train/val/test metrics.
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    if "is_labeled" not in df.columns:
        df["is_labeled"] = df["label"].apply(is_supervised_label_value)


def load_manifest_arrays(
    manifest_csv: str | Path,
    image_size: tuple[int, int],
    cache_npz: Optional[str | Path] = None,
    limit: Optional[int] = None,
    overwrite_cache: bool = False,
    slice_fractions: tuple[float, float, float] = (0.5, 0.5, 0.5),
    slice_indices: Optional[tuple[int, int, int]] = None,
    slices_per_plane: int = 10,
    slice_fraction_min: float = 0.15,
    slice_fraction_max: float = 0.85,
    input_mode: str = "2.5d",
    plane: str = "axial",
    n_slices: int = 15,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read manifest and load/cache images.

    Returns:
        df: manifest dataframe
        X: float32 array. Shape depends on --input-mode:
           2.5d -> (N, 3 * slices_per_plane, H, W)
           2d   -> (N, n_slices, 3, H, W)
        y: int64 labels of shape (N,)
    """
    input_mode = str(input_mode).lower().replace("_", "-")
    if input_mode in {"25d", "2.5d", "2-5d", "2_5d"}:
        input_mode = "2.5d"
    elif input_mode != "2d":
        raise ValueError(f"Unsupported input_mode={input_mode}. Use '2d' or '2.5d'.")
    manifest_csv = Path(manifest_csv)
    df = pd.read_csv(manifest_csv)
    validate_manifest(df)
    if limit is not None and limit > 0:
        # Keep class balance approximately by sampling within label.
        frames = []
        per_class = max(1, limit // max(1, df["label"].nunique()))
        for _, g in df.groupby("label"):
            frames.append(g.sample(n=min(len(g), per_class), random_state=42))
        df = pd.concat(frames, ignore_index=True)
        if len(df) < limit:
            extra = df.index
            remaining = pd.read_csv(manifest_csv).drop(index=extra, errors="ignore")
            if len(remaining):
                df = pd.concat([df, remaining.sample(n=min(len(remaining), limit - len(df)), random_state=43)], ignore_index=True)
        df = df.sample(frac=1.0, random_state=44).reset_index(drop=True)

    cache_npz = Path(cache_npz) if cache_npz else None
    if cache_npz and cache_npz.exists() and not overwrite_cache:
        data = np.load(cache_npz, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        cached_paths = [str(p) for p in data["image_path"].tolist()]
        current_paths = [str(p) for p in df["image_path"].tolist()]
        cached_fracs = tuple(data["slice_fractions"].tolist()) if "slice_fractions" in data.files else None
        cached_indices = tuple(data["slice_indices"].tolist()) if "slice_indices" in data.files else None
        cached_slices_per_plane = int(data["slices_per_plane"]) if "slices_per_plane" in data.files else None
        cached_slice_fraction_min = float(data["slice_fraction_min"]) if "slice_fraction_min" in data.files else None
        cached_slice_fraction_max = float(data["slice_fraction_max"]) if "slice_fraction_max" in data.files else None
        cached_input_mode = str(data["input_mode"].tolist()) if "input_mode" in data.files else "2.5d"
        cached_plane = str(data["plane"].tolist()) if "plane" in data.files else "axial"
        cached_n_slices = int(data["n_slices"]) if "n_slices" in data.files else None
        current_indices = tuple(slice_indices) if slice_indices is not None else (-1, -1, -1)
        slice_config_ok = (
            cached_fracs == tuple(slice_fractions)
            and cached_indices == current_indices
            and cached_slices_per_plane == int(slices_per_plane)
            and np.isclose(cached_slice_fraction_min, float(slice_fraction_min))
            and np.isclose(cached_slice_fraction_max, float(slice_fraction_max))
            and cached_input_mode == input_mode
            and cached_plane == str(plane).lower()
            and cached_n_slices == int(n_slices)
        )
        if len(cached_paths) == len(current_paths) and cached_paths == current_paths and slice_config_ok:
            return df, X, y
        print("Cache exists but does not match current manifest/limit/slice setting. Rebuilding cache.")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    keep_rows = []
    desc = "load MRI 2D slice inputs" if input_mode == "2d" else "load MRI 2.5D multi-slice inputs"
    for i, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        path = Path(str(row["image_path"]))
        if not path.exists():
            print(f"[WARN] Missing image: {path}")
            continue
        try:
            if input_mode == "2d":
                X_item = load_mri_2d_slices(
                    path,
                    image_size=image_size,
                    plane=plane,
                    n_slices=n_slices,
                    slice_fraction_min=slice_fraction_min,
                    slice_fraction_max=slice_fraction_max,
                )
            else:
                X_item = load_mri_multi_slices(
                    path,
                    image_size=image_size,
                    slices_per_plane=slices_per_plane,
                    slice_fraction_min=slice_fraction_min,
                    slice_fraction_max=slice_fraction_max,
                    slice_fractions=slice_fractions,
                    slice_indices=slice_indices,
                )
            X_list.append(X_item)
            y_list.append(int(row["label"]))
            keep_rows.append(i)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")

    if not X_list:
        raise RuntimeError("No images could be loaded from the manifest.")
    df = df.loc[keep_rows].reset_index(drop=True)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    if cache_npz:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        cache_indices = tuple(slice_indices) if slice_indices is not None else (-1, -1, -1)
        np.savez_compressed(
            cache_npz,
            X=X,
            y=y,
            image_path=df["image_path"].astype(str).to_numpy(),
            slice_fractions=np.asarray(slice_fractions, dtype=np.float32),
            slice_indices=np.asarray(cache_indices, dtype=np.int64),
            slices_per_plane=np.asarray(int(slices_per_plane), dtype=np.int64),
            slice_fraction_min=np.asarray(float(slice_fraction_min), dtype=np.float32),
            slice_fraction_max=np.asarray(float(slice_fraction_max), dtype=np.float32),
            input_mode=np.asarray(input_mode),
            plane=np.asarray(str(plane).lower()),
            n_slices=np.asarray(int(n_slices), dtype=np.int64),
        )
        print(f"Saved cache: {cache_npz}")
    return df, X, y


# -----------------------------
# Losses and models
# -----------------------------


class FocalLoss(nn.Module):
    """Multi-class focal loss on logits."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha.float() if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce).clamp(1e-7, 1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


class CNNClassifier2D(nn.Module):
    """Small multi-stack 2D CNN feature extractor + classifier head."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        cnn_stacks: int = 3,
        base_filters: int = 32,
        dense_units: int = 128,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        act_layer = activation_module_factory(activation)

        blocks = []
        ch = in_channels
        filters = base_filters
        for _ in range(cnn_stacks):
            blocks.extend(
                [
                    nn.Conv2d(ch, filters, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(filters),
                    make_activation(act_layer),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            ch = filters
            filters *= 2
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, dense_units),
            make_activation(act_layer),
        )
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(dense_units, num_classes)
        self.feature_dim = dense_units
        self.backbone_name = "current"

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.backbone(x)
        x = self.pool(x)
        feat = self.feature(x)
        logits = self.classifier(self.dropout(feat))
        if return_features:
            return logits, feat
        return logits


class BasicCNNBaseline2D(nn.Module):
    """Basic 2D baseline from train_all_2d.py, adapted to arbitrary input channels.

    The uploaded baseline uses a 3-channel 2D slice input. In this AL/DQN script
    the classifier receives the existing 2.5D tensor, usually
    3 * slices_per_plane channels. Therefore the first convolution accepts
    in_channels directly, while the rest of the baseline is kept the same.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=False),
        )
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(256, num_classes)
        self.feature_dim = 256
        self.backbone_name = "basic_cnn"

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.features(x)
        x = self.pool(x)
        feat = self.proj(x)
        logits = self.classifier(self.dropout(feat))
        if return_features:
            return logits, feat
        return logits


def _resnet18_weights(pretrained: bool):
    if not pretrained or models is None:
        return None
    try:
        return models.ResNet18_Weights.IMAGENET1K_V1
    except AttributeError:
        return "IMAGENET1K_V1"


def _densenet121_weights(pretrained: bool):
    if not pretrained or models is None:
        return None
    try:
        return models.DenseNet121_Weights.IMAGENET1K_V1
    except AttributeError:
        return "IMAGENET1K_V1"


def _adapt_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Replace a 3-channel torchvision stem conv with an arbitrary-channel conv."""
    in_channels = int(in_channels)
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        old_w = conv.weight.detach()
        if old_w.ndim == 4 and old_w.shape[1] > 0:
            # MRI channels are slices rather than RGB. Average the pretrained RGB
            # filters and replicate to all input slices to keep activation scale stable.
            mean_w = old_w.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_w.repeat(1, in_channels, 1, 1))
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias.detach())
    return new_conv


class TorchvisionBackboneClassifier2D(nn.Module):
    """ResNet18/DenseNet121 classifier wrapper with return_features support for DQN.

    DQN state construction calls model(x, return_features=True), so torchvision
    models are wrapped to expose a stable feature vector before the final linear
    classifier. The first convolution is adapted from RGB to the current 2.5D
    channel count.
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        num_classes: int = 2,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        if models is None:
            raise RuntimeError("torchvision is required for resnet18/densenet121 baselines.")
        name = str(name).lower()
        self.backbone_name = name
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        try:
            if name == "resnet18":
                m = models.resnet18(weights=_resnet18_weights(pretrained))
                m.conv1 = _adapt_conv_in_channels(m.conv1, in_channels)
                self.feature_dim = int(m.fc.in_features)
                m.fc = nn.Identity()
                self.backbone = m
                self.classifier = nn.Linear(self.feature_dim, num_classes)
            elif name == "densenet121":
                m = models.densenet121(weights=_densenet121_weights(pretrained))
                m.features.conv0 = _adapt_conv_in_channels(m.features.conv0, in_channels)
                self.feature_dim = int(m.classifier.in_features)
                self.features = m.features
                self.classifier = nn.Linear(self.feature_dim, num_classes)
            else:
                raise ValueError(f"Unknown torchvision baseline: {name}")
        except Exception as e:
            if pretrained:
                print(f"[WARN] Could not load pretrained {name} weights ({e}). Falling back to random initialization.")
                self.__init__(name=name, in_channels=in_channels, num_classes=num_classes, dropout=dropout, pretrained=False)
                return
            raise

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if self.backbone_name == "resnet18":
            feat = self.backbone(x)
        elif self.backbone_name == "densenet121":
            feat_map = self.features(x)
            feat_map = F.relu(feat_map, inplace=False)
            feat = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)
        else:
            raise ValueError(self.backbone_name)
        logits = self.classifier(self.dropout(feat))
        if return_features:
            return logits, feat
        return logits

def activation_module_factory(activation: str) -> type[nn.Module]:
    name = str(activation).lower().replace("-", "_")
    if name == "leaky_relu":
        return nn.LeakyReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid
    return nn.ReLU


def make_activation(act_layer: type[nn.Module]) -> nn.Module:
    if act_layer in {nn.ReLU, nn.LeakyReLU}:
        return act_layer(inplace=True)
    return act_layer()


class DQN(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, mlp_layers: int = 2, num_actions: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(max(1, mlp_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass
class DQNConfig:
    state_dim: int
    hidden_dim: int = 128
    mlp_layers: int = 4
    gamma: float = 0.95
    epsilon_start: float = 0.8
    epsilon_final: float = 0.05
    epsilon_decay: float = 0.995
    lr: float = 1e-3
    scope_alpha: float = 0.2
    scope_weight: float = 0.05
    q_policy_temperature: float = 1.0
    advantage_clip: float = 5.0
    dqn_grad_clip: float = 5.0
    replay_size: int = 4096
    replay_batch_size: int = 64


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.data = deque(maxlen=maxlen)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.data.append((state.astype(np.float32), int(action), float(reward), next_state.astype(np.float32), bool(done)))

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int):
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )


class DQNAgent:
    def __init__(self, cfg: DQNConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.epsilon = cfg.epsilon_start
        self.model = DQN(cfg.state_dim, cfg.hidden_dim, cfg.mlp_layers).to(device)
        self.target = DQN(cfg.state_dim, cfg.hidden_dim, cfg.mlp_layers).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.replay_size)
        self.steps = 0

    @torch.no_grad()
    def act(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        if states.ndim == 1:
            states = states[None, :]
        if not greedy:
            random_mask = np.random.rand(len(states)) < self.epsilon
        else:
            random_mask = np.zeros(len(states), dtype=bool)
        st = torch.from_numpy(states.astype(np.float32)).to(self.device)
        q = self.model(st).detach().cpu().numpy()
        actions = np.argmax(q, axis=1).astype(np.int64)
        actions[random_mask] = np.random.randint(0, 2, size=int(random_mask.sum()))
        return actions

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, updates: int = 1) -> dict[str, float]:
        if len(self.buffer) == 0:
            return {"dqn_loss": float("nan")}
        losses = []
        td_losses = []
        scope_losses = []
        advantage_means = []
        advantage_stds = []
        advantage_abs_maxes = []
        for _ in range(updates):
            states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.replay_batch_size)
            states_t = torch.from_numpy(states).to(self.device)
            actions_t = torch.from_numpy(actions).to(self.device)
            rewards_t = torch.from_numpy(rewards).to(self.device)
            next_t = torch.from_numpy(next_states).to(self.device)
            dones_t = torch.from_numpy(dones).to(self.device)

            q = self.model(states_t)
            q_sa = q.gather(1, actions_t[:, None]).squeeze(1)
            with torch.no_grad():
                q_next = self.target(next_t).max(dim=1).values
                target = rewards_t + self.cfg.gamma * q_next * (1.0 - dones_t)

            # Main DQN loss. Huber is intentionally used instead of MSE to reduce
            # sensitivity to occasional large rewards/targets.
            td_loss = F.smooth_l1_loss(q_sa, target)

            # Paper-style Scope Loss adapted to a DQN policy induced by Q-values.
            # The paper discusses Scope Loss as (A - alpha * p_i) * log(p_i),
            # and recommends z-score batch normalization of the advantage term A
            # so that alpha does not need environment-specific manual tuning.
            #
            # We therefore do NOT use raw TD error directly as the scope term.
            # Instead, use a detached advantage proxy:
            #     A = td_target - V(s),  V(s)=sum_a pi(a|s) Q(s,a),
            # then z-score and clip it within the replay mini-batch.
            temperature = max(float(self.cfg.q_policy_temperature), 1e-6)
            policy = torch.softmax(q / temperature, dim=1)
            p_action = policy.gather(1, actions_t[:, None]).squeeze(1).clamp(1e-7, 1.0 - 1e-7)
            log_p_action = torch.log(p_action)

            with torch.no_grad():
                policy_detached = policy.detach()
                v_s = (policy_detached * q.detach()).sum(dim=1)
                advantage = target.detach() - v_s
                adv_mean = advantage.mean()
                adv_std = advantage.std(unbiased=False)
                if torch.isfinite(adv_std) and float(adv_std.detach().cpu()) > 1e-8:
                    advantage_z = (advantage - adv_mean) / (adv_std + 1e-8)
                else:
                    advantage_z = torch.zeros_like(advantage)
                advantage_z = advantage_z.clamp(-self.cfg.advantage_clip, self.cfg.advantage_clip)

            scope_loss = ((advantage_z - self.cfg.scope_alpha * p_action) * log_p_action).mean()

            loss = td_loss + self.cfg.scope_weight * scope_loss
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.dqn_grad_clip)
            self.opt.step()

            self.steps += 1
            if self.steps % 50 == 0:
                self.target.load_state_dict(self.model.state_dict())
            self.epsilon = max(self.cfg.epsilon_final, self.epsilon * self.cfg.epsilon_decay)
            losses.append(float(loss.detach().cpu()))
            td_losses.append(float(td_loss.detach().cpu()))
            scope_losses.append(float(scope_loss.detach().cpu()))
            advantage_means.append(float(advantage.mean().detach().cpu()))
            advantage_stds.append(float(advantage.std(unbiased=False).detach().cpu()))
            advantage_abs_maxes.append(float(advantage.abs().max().detach().cpu()))
        return {
            "dqn_loss": float(np.mean(losses)),
            "td_loss": float(np.mean(td_losses)),
            "scope_loss": float(np.mean(scope_losses)),
            "advantage_mean": float(np.mean(advantage_means)),
            "advantage_std": float(np.mean(advantage_stds)),
            "advantage_abs_max": float(np.mean(advantage_abs_maxes)),
            "epsilon": float(self.epsilon),
        }




# -----------------------------
# Classifier training helpers
# -----------------------------


def class_weights_for_labels(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)



class SliceAggregateClassifier(nn.Module):
    """Wrap a 2D backbone so it can receive (B, S, 3, H, W) MRI inputs.

    The backbone is applied to every 2D slice. Slice logits and features are
    averaged to one MRI-level prediction/feature vector. This keeps active
    learning, DQN state construction, and final metrics at MRI/patient level
    while using true 2D CNN backbones.
    """

    def __init__(self, base: nn.Module, aggregation: str = "mean_logits"):
        super().__init__()
        self.base = base
        self.aggregation = str(aggregation)
        self.feature_dim = getattr(base, "feature_dim", None)
        self.backbone_name = f"2d_{getattr(base, 'backbone_name', base.__class__.__name__)}"

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.ndim != 5:
            return self.base(x, return_features=return_features)
        b, s, c, h, w = x.shape
        flat = x.reshape(b * s, c, h, w)
        logits_flat, feat_flat = self.base(flat, return_features=True)
        logits_s = logits_flat.reshape(b, s, -1)
        feat_s = feat_flat.reshape(b, s, -1)
        # Mean logits is stable for training with CE/focal loss. It is also close
        # to patient-level averaging while preserving differentiability.
        logits = logits_s.mean(dim=1)
        feat = feat_s.mean(dim=1)
        if return_features:
            return logits, feat
        return logits

def make_classifier(args, device: torch.device) -> nn.Module:
    """Build the classifier backbone used by the AL/DQN loop.

    --cnn-arch current      : original compact 2.5D CNN in this script.
    --cnn-arch basic_cnn    : BasicCNN baseline from train_all_2d.py, adapted to 2.5D channels.
    --cnn-arch resnet18     : torchvision ResNet18 baseline, stem adapted to 2.5D channels.
    --cnn-arch densenet121  : torchvision DenseNet121 baseline, stem adapted to 2.5D channels.
    """
    input_mode = str(getattr(args, "input_mode", "2.5d")).lower()
    in_channels = 3 if input_mode == "2d" else max(1, int(args.slices_per_plane)) * 3
    arch = str(getattr(args, "cnn_arch", "current")).lower()
    if arch == "current":
        model = CNNClassifier2D(
            in_channels=in_channels,
            num_classes=2,
            cnn_stacks=args.cnn_stacks,
            base_filters=args.base_filters,
            dense_units=args.dense_units,
            dropout=args.dropout,
            activation=args.activation,
        )
    elif arch == "basic_cnn":
        model = BasicCNNBaseline2D(
            in_channels=in_channels,
            num_classes=2,
            dropout=max(float(args.dropout), 0.0),
        )
    elif arch in {"resnet18", "densenet121"}:
        model = TorchvisionBackboneClassifier2D(
            name=arch,
            in_channels=in_channels,
            num_classes=2,
            dropout=max(float(args.dropout), 0.0),
            pretrained=bool(getattr(args, "pretrained_backbone", True)),
        )
    else:
        raise ValueError(f"Unknown --cnn-arch: {arch}")
    if input_mode == "2d":
        model = SliceAggregateClassifier(model, aggregation=str(getattr(args, "slice_aggregation", "mean_logits")))
    print(f"  Classifier backbone: {arch} | input_mode={input_mode} | input_channels={in_channels} | feature_dim={getattr(model, 'feature_dim', 'unknown')}")
    return model.to(device)

def make_classifier_optimizer(model: nn.Module, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def batch_predict(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    feats = []
    loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits, feat = model(xb, return_features=True)
            prob = torch.softmax(logits, dim=1)
            probs.append(prob.cpu().numpy())
            feats.append(feat.cpu().numpy())
    return np.concatenate(probs, axis=0), np.concatenate(feats, axis=0)


def compute_classifier_loss(
    model: CNNClassifier2D,
    X: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
    amp: bool = False,
) -> float:
    if len(X) == 0:
        return float("nan")
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y.astype(np.int64)).long())
    loader = DataLoader(dataset, batch_size=min(batch_size, max(1, len(X))), shuffle=False, drop_last=False)
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            bs = xb.size(0)
            total += float(loss.detach().cpu()) * bs
            n += bs
    return float(total / max(n, 1))


def fit_classifier(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    args,
    device: torch.device,
    epochs: int,
    verbose: bool = True,
    val_X: Optional[np.ndarray] = None,
    val_y: Optional[np.ndarray] = None,
    epoch_history: Optional[list[dict[str, Any]]] = None,
    fold_dir: Optional[Path] = None,
    cycle: int = 0,
    stage: str = "train",
    labeled_count: Optional[int] = None,
) -> dict[str, float]:
    if len(X) == 0:
        raise ValueError("Cannot train classifier with no labeled samples.")
    batch_size = min(args.batch_size, max(1, len(X)))
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y.astype(np.int64)).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    weights = class_weights_for_labels(y, device)
    if args.loss_name == "ce":
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=weights)
    opt = make_classifier_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    last_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = xb.size(0)
            total += float(loss.detach().cpu()) * bs
            n += bs
        last_loss = total / max(n, 1)

        # Optional per-epoch metrics, matching the style of the SFCN training script.
        if getattr(args, "save_epoch_metrics", True) and epoch_history is not None:
            train_metrics, train_pred = evaluate_classifier(model, X, y, args.batch_size, device, threshold=0.5)
            train_metrics["loss"] = float(last_loss)
            val_metrics = {}
            if val_X is not None and val_y is not None and len(val_X) > 0:
                val_metrics, _ = evaluate_classifier(model, val_X, val_y, args.batch_size, device, threshold=0.5)
                val_metrics["loss"] = compute_classifier_loss(model, val_X, val_y, criterion, args.batch_size, device, amp=args.amp)
                val_thr = find_optimal_threshold(val_y, evaluate_classifier(model, val_X, val_y, args.batch_size, device, threshold=0.5)[1]["prob_demented"].to_numpy())
                val_metrics["optimal_threshold"] = float(val_thr)
            row = {
                "global_epoch": len(epoch_history) + 1,
                "cycle": int(cycle),
                "stage": stage,
                "epoch_in_stage": int(epoch),
                "epochs_in_stage": int(epochs),
                "labeled_count": int(labeled_count if labeled_count is not None else len(X)),
            }
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            epoch_history.append(row)
            if fold_dir is not None:
                pd.DataFrame(epoch_history).to_csv(fold_dir / "epoch_metrics.csv", index=False)

        if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0):
            print(f"    classifier epoch {epoch}/{epochs}, loss={last_loss:.4f}")
    return {"loss": float(last_loss)}


def evaluate_classifier(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    probs, _ = batch_predict(model, X, batch_size=batch_size, device=device)
    y_prob = probs[:, 1]
    metrics = compute_binary_metrics(y, y_prob, threshold=threshold)
    pred_df = pd.DataFrame({"y_true": y.astype(int), "prob_demented": y_prob, "y_pred": (y_prob >= threshold).astype(int)})
    return metrics, pred_df


# -----------------------------
# Active learning
# -----------------------------


@dataclass
class ActiveLearningResult:
    history: list[dict[str, Any]]
    selected_rows: list[dict[str, Any]]
    best_val_threshold: float
    labeled_count: int
    best_cycle: int = 0
    best_val_score: float = float("nan")
    final_labeled_count: int = 0
    final_queried_count: int = 0


class ActiveLearningRunner:
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device

    @staticmethod
    def entropy(probs: np.ndarray) -> np.ndarray:
        p = np.clip(probs, 1e-9, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    @staticmethod
    def centroid_entropy(
        unlabeled_feats: np.ndarray,
        labeled_feats: np.ndarray,
        labeled_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Paper-like uncertainty score for the non-annotation reward.

        The paper defines the action=0 reward through an entropy term computed
        from distances between each unlabeled feature vector and the labeled
        class centroids. This function implements a numerically stable version:

            p(class=j | x) = softmax(-||f_x - m_j||^2 / temperature)
            L(x) = -sum_j p_j log(p_j)

        The entropy is normalized to [0, 1] for the binary case.
        High value = uncertain / close to both class centroids.
        Low value  = confident / close to one class centroid.
        """
        labeled_y = np.asarray(labeled_y).astype(int)
        classes = np.array([0, 1], dtype=int)
        if len(labeled_feats) == 0:
            ent = np.ones(len(unlabeled_feats), dtype=np.float32)
            probs = np.full((len(unlabeled_feats), 2), 0.5, dtype=np.float32)
            d2 = np.full((len(unlabeled_feats), 2), np.nan, dtype=np.float32)
            return ent, probs, d2

        global_mean = labeled_feats.mean(axis=0)
        centroids = []
        for c in classes:
            mask = labeled_y == c
            centroids.append(labeled_feats[mask].mean(axis=0) if np.any(mask) else global_mean)
        centroids = np.stack(centroids).astype(np.float32)

        d2 = ((unlabeled_feats[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
        finite = d2[np.isfinite(d2)]
        temperature = float(np.median(finite)) if finite.size else 1.0
        if not np.isfinite(temperature) or temperature <= 1e-8:
            temperature = 1.0
        logits = -d2 / temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        centroid_probs = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
        ent = -np.sum(centroid_probs * np.log(np.clip(centroid_probs, 1e-12, 1.0)), axis=1)
        ent = ent / np.log(len(classes))
        return ent.astype(np.float32), centroid_probs.astype(np.float32), d2.astype(np.float32)

    def _initial_labeled_indices(self, train_indices: np.ndarray, y_all: np.ndarray) -> tuple[list[int], list[int]]:
        frac = self.args.initial_label_fraction
        labels = y_all[train_indices]
        n_init = max(2, int(round(len(train_indices) * frac)))
        n_init = min(n_init, len(train_indices))
        try:
            if len(np.unique(labels)) == 2 and min(np.bincount(labels, minlength=2)) >= 2:
                init_idx, rest_idx = train_test_split(
                    train_indices,
                    train_size=n_init,
                    random_state=self.args.seed,
                    stratify=labels,
                )
            else:
                init_idx, rest_idx = train_test_split(train_indices, train_size=n_init, random_state=self.args.seed)
        except Exception:
            rng = np.random.default_rng(self.args.seed)
            perm = rng.permutation(train_indices)
            init_idx, rest_idx = perm[:n_init], perm[n_init:]
        return list(map(int, init_idx)), list(map(int, rest_idx))

    def run(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        df: pd.DataFrame,
        fold_dir: Path,
        extra_unlabeled_indices: Optional[np.ndarray] = None,
    ) -> tuple[nn.Module, ActiveLearningResult, dict[str, float], pd.DataFrame]:
        args = self.args
        fold_dir.mkdir(parents=True, exist_ok=True)
        labeled_idx, unlabeled_idx = self._initial_labeled_indices(train_indices, y_all)
        if extra_unlabeled_indices is not None and len(extra_unlabeled_indices):
            # External unlabeled rows (e.g. OASIS1 missing CDR) enter the AL pool.
            # When selected/query-revealed, empty/NaN/-1 labels are imputed as
            # class 0 and then added to supervised training.
            extra = [int(i) for i in np.asarray(extra_unlabeled_indices, dtype=int).tolist()]
            blocked = set(map(int, labeled_idx)) | set(map(int, val_indices)) | set(map(int, test_indices))
            seen = set(map(int, unlabeled_idx))
            for i in extra:
                if i not in blocked and i not in seen:
                    unlabeled_idx.append(i)
                    seen.add(i)
        selected_rows: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        epoch_history: list[dict[str, Any]] = []

        model = make_classifier(args, self.device)
        print(f"  Initial labeled images: {len(labeled_idx)} / train pool {len(train_indices)}")
        fit_classifier(
            model,
            X_all[labeled_idx],
            y_all[labeled_idx],
            args,
            self.device,
            epochs=args.initial_epochs,
            verbose=args.verbose,
            val_X=X_all[val_indices],
            val_y=y_all[val_indices],
            epoch_history=epoch_history,
            fold_dir=fold_dir,
            cycle=0,
            stage="initial",
            labeled_count=len(labeled_idx),
        )

        val_metrics, val_pred = evaluate_classifier(model, X_all[val_indices], y_all[val_indices], args.batch_size, self.device, threshold=0.5)
        best_val_threshold = find_optimal_threshold(y_all[val_indices], val_pred["prob_demented"].to_numpy()) if args.use_val_optimal_threshold else 0.5
        val_metrics_opt, _ = evaluate_classifier(model, X_all[val_indices], y_all[val_indices], args.batch_size, self.device, threshold=best_val_threshold)
        train_metrics_opt, _ = evaluate_classifier(model, X_all[train_indices], y_all[train_indices], args.batch_size, self.device, threshold=best_val_threshold)
        history.append({
            "cycle": 0,
            "labeled_count": len(labeled_idx),
            "queried": 0,
            **{f"train_{k}": v for k, v in train_metrics_opt.items()},
            **{f"val_{k}": v for k, v in val_metrics_opt.items()},
        })

        best_checkpoint_metric = getattr(args, "best_checkpoint_metric", "composite")
        best_val_score = checkpoint_score(val_metrics_opt, best_checkpoint_metric)
        best_cycle = 0
        best_labeled_count = len(labeled_idx)
        best_val_threshold_for_checkpoint = float(best_val_threshold)
        best_model_state = clone_state_dict(model)
        best_policy_state = None

        # Build the DQN active-learning policy after we know the classifier feature dim.
        probs1, feats1 = batch_predict(model, X_all[labeled_idx[:1]], batch_size=1, device=self.device)
        state_dim = feats1.shape[1] + probs1.shape[1]
        dqn_cfg = DQNConfig(
            state_dim=state_dim,
            hidden_dim=args.q_hidden_dim,
            mlp_layers=args.q_mlp_layers,
            gamma=args.dqn_gamma,
            epsilon_start=args.epsilon_start,
            epsilon_final=args.epsilon_final,
            epsilon_decay=args.epsilon_decay,
            lr=args.dqn_lr,
            scope_alpha=args.scope_alpha,
            scope_weight=args.scope_weight,
            q_policy_temperature=args.q_policy_temperature,
            advantage_clip=args.advantage_clip,
            dqn_grad_clip=args.dqn_grad_clip,
            replay_size=args.replay_size,
            replay_batch_size=args.replay_batch_size,
        )
        policy_agent = DQNAgent(dqn_cfg, self.device)
        best_policy_state = clone_state_dict(policy_agent.model)

        # Active-learning stopping rule.
        # Default: run a fixed number of cycles, e.g. --cycles 5.
        # Paper-like: use --until-unlabeled-exhausted to keep looping until X_U is empty,
        # optionally capped by --max-annotations. This matches Algorithm 1's stopping rule:
        # stop when all unlabeled images are exhausted or the annotation limit is reached.
        total_queried = 0
        cycle = 0
        while True:
            cycle += 1
            if len(unlabeled_idx) == 0:
                print("  Unlabeled pool exhausted.")
                break
            if (not args.until_unlabeled_exhausted) and cycle > args.cycles:
                break
            if args.max_annotations >= 0 and total_queried >= args.max_annotations:
                print(f"  Annotation budget reached: {total_queried}/{args.max_annotations}.")
                break

            cycle_limit_text = "until exhausted" if args.until_unlabeled_exhausted else str(args.cycles)
            print(f"\n  AL cycle {cycle}/{cycle_limit_text}: unlabeled={len(unlabeled_idx)}, labeled={len(labeled_idx)}, queried_total={total_queried}")
            unlabeled_arr = np.asarray(unlabeled_idx, dtype=int)
            probs, feats = batch_predict(model, X_all[unlabeled_arr], batch_size=args.inference_batch_size, device=self.device)
            states = np.concatenate([feats, probs], axis=1).astype(np.float32)
            softmax_ent = self.entropy(probs)
            if args.reward_mode == "centroid":
                labeled_arr_for_reward = np.asarray(labeled_idx, dtype=int)
                _, labeled_feats_for_reward = batch_predict(
                    model,
                    X_all[labeled_arr_for_reward],
                    batch_size=args.inference_batch_size,
                    device=self.device,
                )
                ent, centroid_probs, centroid_d2 = self.centroid_entropy(
                    feats, labeled_feats_for_reward, y_all[labeled_arr_for_reward]
                )
            else:
                ent = softmax_ent
                centroid_probs = np.full((len(unlabeled_arr), 2), np.nan, dtype=np.float32)
                centroid_d2 = np.full((len(unlabeled_arr), 2), np.nan, dtype=np.float32)

            # Dynamic threshold, increases as validation accuracy improves.
            # High entropy = uncertain. A larger threshold lets the agent skip more
            # samples once the classifier becomes reliable, matching the paper's
            # description that ∂ increases during training.
            val_acc = float(val_metrics_opt.get("accuracy", 0.5))
            entropy_threshold = float(np.quantile(ent, min(0.9, max(0.1, 0.35 + 0.5 * val_acc))))
            actions = policy_agent.act(states, greedy=False)
            # DQN action sampling is epsilon-greedy, but budgeted selection must not
            # truncate by arbitrary pool order. Rank query candidates by Q(label)-Q(skip).
            with torch.no_grad():
                st_rank = torch.from_numpy(states.astype(np.float32)).to(self.device)
                q_rank = policy_agent.model(st_rank).detach().cpu().numpy()
            label_scores = (q_rank[:, 1] - q_rank[:, 0]).astype(np.float32)
            q_shift = q_rank - q_rank.max(axis=1, keepdims=True)
            q_exp = np.exp(q_shift)
            action_probs = (q_exp / np.clip(q_exp.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)

            selected_mask = actions == 1
            selected_local = np.where(selected_mask)[0].tolist()

            # Safety fallback: if the policy is too conservative, query the highest-entropy samples.
            # If the policy selects more than the budget, keep the strongest label-action scores
            # instead of the first indices in pool order. This makes DQN selection meaningful
            # and improves reproducibility of selected_samples.
            budget = min(args.budget_per_cycle, len(unlabeled_idx))
            if args.max_annotations >= 0:
                budget = min(budget, max(0, args.max_annotations - total_queried))
            if budget <= 0:
                print("    No remaining annotation budget for this cycle.")
                break

            selected_local = sorted(selected_local, key=lambda j: float(label_scores[j]), reverse=True)
            if len(selected_local) < budget:
                # Fill missing query slots using the paper-like entropy score.
                ranked = np.argsort(-ent).tolist()
                seen = set(selected_local)
                for j in ranked:
                    if j not in seen:
                        selected_local.append(j)
                        seen.add(j)
                    if len(selected_local) >= budget:
                        break
            selected_local = selected_local[:budget]
            selected_global = unlabeled_arr[selected_local].tolist()

            # Store contextual-bandit style transitions. Done=True makes DQN targets immediate rewards.
            selected_set = set(selected_local)
            for local_i in range(len(unlabeled_arr)):
                action = 1 if local_i in selected_set else int(actions[local_i])
                if action == 1:
                    reward = -float(args.label_cost)
                else:
                    # Reward skipping only if the sample is confident enough.
                    reward = 1.0 if ent[local_i] < entropy_threshold else -1.0
                policy_agent.push(states[local_i], action, reward, states[local_i], True)

            agent_stats = policy_agent.update(updates=args.dqn_updates_per_cycle)

            revealed_selected_global: list[int] = []
            external_unlabeled_selected = 0
            for rank, local_i in enumerate(selected_local, start=1):
                idx = int(unlabeled_arr[local_i])
                row = df.iloc[idx].to_dict()
                raw_label_before_reveal = y_all[idx]
                revealed_label, label_imputed_from_empty = reveal_label_or_zero(raw_label_before_reveal)

                # Once queried, the oracle/reveal step must produce a supervised
                # class. For empty OASIS1 unlabeled labels, treat the revealed
                # result as class 0 instead of leaving it blank/-1.
                y_all[idx] = int(revealed_label)
                if "label" in df.columns:
                    df.at[idx, "label"] = int(revealed_label)
                if "label_name" in df.columns:
                    df.at[idx, "label_name"] = "demented" if int(revealed_label) == 1 else "non_demented"
                if "is_labeled" in df.columns:
                    df.at[idx, "is_labeled"] = True
                if "pool" in df.columns:
                    df.at[idx, "pool"] = "labeled_after_query"
                if "split" in df.columns:
                    df.at[idx, "split"] = f"queried_cycle_{cycle}"

                revealed_selected_global.append(idx)
                if label_imputed_from_empty:
                    external_unlabeled_selected += 1
                selected_rows.append(
                    {
                        "cycle": cycle,
                        "rank": rank,
                        "manifest_index": idx,
                        "label": int(revealed_label),
                        "raw_label_before_reveal": str(raw_label_before_reveal),
                        "label_available": True,
                        "label_imputed_from_empty": bool(label_imputed_from_empty),
                        "external_unlabeled": bool(label_imputed_from_empty),
                        "entropy": float(ent[local_i]),
                        "reward_entropy": float(ent[local_i]),
                        "softmax_entropy": float(softmax_ent[local_i]),
                        "centroid_prob_class0": float(centroid_probs[local_i, 0]),
                        "centroid_prob_class1": float(centroid_probs[local_i, 1]),
                        "centroid_d2_class0": float(centroid_d2[local_i, 0]),
                        "centroid_d2_class1": float(centroid_d2[local_i, 1]),
                        "prob_demented": float(probs[local_i, 1]),
                        "label_score": float(label_scores[local_i]),
                        "action": 1,
                        "reward_mode": args.reward_mode,
                        "image_path": row.get("image_path", ""),
                        "subject_key": row.get("subject_key", ""),
                        "mri_id": row.get("mri_id", ""),
                        "dataset": row.get("dataset", ""),
                    }
                )

            labeled_idx.extend(revealed_selected_global)
            total_queried += len(selected_global)
            selected_global_set = set(selected_global)
            unlabeled_idx = [int(i) for i in unlabeled_idx if int(i) not in selected_global_set]

            if args.reset_classifier_each_cycle:
                model = make_classifier(args, self.device)
            fit_classifier(
                model,
                X_all[labeled_idx],
                y_all[labeled_idx],
                args,
                self.device,
                epochs=args.retrain_epochs,
                verbose=args.verbose,
                val_X=X_all[val_indices],
                val_y=y_all[val_indices],
                epoch_history=epoch_history,
                fold_dir=fold_dir,
                cycle=cycle,
                stage=f"retrain_cycle_{cycle}",
                labeled_count=len(labeled_idx),
            )

            val_metrics_05, val_pred = evaluate_classifier(model, X_all[val_indices], y_all[val_indices], args.batch_size, self.device, threshold=0.5)
            best_val_threshold = find_optimal_threshold(y_all[val_indices], val_pred["prob_demented"].to_numpy()) if args.use_val_optimal_threshold else 0.5
            val_metrics_opt, _ = evaluate_classifier(model, X_all[val_indices], y_all[val_indices], args.batch_size, self.device, threshold=best_val_threshold)
            train_metrics_opt, _ = evaluate_classifier(model, X_all[train_indices], y_all[train_indices], args.batch_size, self.device, threshold=best_val_threshold)
            cycle_row = {
                "cycle": cycle,
                "labeled_count": len(labeled_idx),
                "queried": len(selected_global),
                "queried_with_labels": len(revealed_selected_global),
                "queried_original_labels": int(len(revealed_selected_global) - external_unlabeled_selected),
                "queried_empty_imputed_as_zero": int(external_unlabeled_selected),
                "queried_external_unlabeled": int(external_unlabeled_selected),
                "total_queried": total_queried,
                "remaining_unlabeled": len(unlabeled_idx),
                "max_annotations": args.max_annotations,
                "until_unlabeled_exhausted": bool(args.until_unlabeled_exhausted),
                "entropy_threshold": entropy_threshold,
                "reward_mode": args.reward_mode,
                "mean_reward_entropy_unlabeled": float(np.mean(ent)) if len(ent) else float("nan"),
                "mean_softmax_entropy_unlabeled": float(np.mean(softmax_ent)) if len(softmax_ent) else float("nan"),
                **{f"dqn_{k}": v for k, v in agent_stats.items()},
                **{f"train_{k}": v for k, v in train_metrics_opt.items()},
                **{f"val_{k}": v for k, v in val_metrics_opt.items()},
                "val_auc_raw_threshold_0_5": val_metrics_05.get("auc", float("nan")),
                "val_f1_threshold_0_5": val_metrics_05.get("f1", float("nan")),
            }
            cycle_score = checkpoint_score(val_metrics_opt, best_checkpoint_metric)
            cycle_row["best_checkpoint_metric"] = best_checkpoint_metric
            cycle_row["checkpoint_score"] = float(cycle_score)
            if np.isfinite(cycle_score) and cycle_score > best_val_score:
                best_val_score = float(cycle_score)
                best_cycle = int(cycle)
                best_labeled_count = len(labeled_idx)
                best_val_threshold_for_checkpoint = float(best_val_threshold)
                best_model_state = clone_state_dict(model)
                best_policy_state = clone_state_dict(policy_agent.model)

            history.append(cycle_row)
            pd.DataFrame(history).to_csv(fold_dir / "cycle_metrics.csv", index=False)
            pd.DataFrame(selected_rows).to_csv(fold_dir / "selected_samples.csv", index=False)
            agent_status = f"eps={policy_agent.epsilon:.3f}"
            print(
                f"    queried={len(selected_global)}, total_queried={total_queried}, "
                f"remaining_unlabeled={len(unlabeled_idx)}, labeled={len(labeled_idx)}, "
                f"val_auc={val_metrics_opt['auc']:.4f}, val_f1={val_metrics_opt['f1']:.4f}, "
                f"threshold={best_val_threshold:.3f}, {agent_status}"
            )

        figures_dir = fold_dir / "figures"
        plot_cycle_metrics(history, figures_dir)
        plot_epoch_metrics(epoch_history, figures_dir)
        plot_selected_samples(selected_rows, figures_dir)

        # Keep the last states for debugging, but use the best validation checkpoint
        # for all final train/val/test outputs. This avoids reporting the last AL
        # cycle when validation performance peaked earlier.
        torch.save(model.state_dict(), fold_dir / "classifier_last.pt")
        torch.save(policy_agent.model.state_dict(), fold_dir / "dqn_last.pt")
        load_state_dict_cpu_safe(model, best_model_state, self.device)
        load_state_dict_cpu_safe(policy_agent.model, best_policy_state, self.device)
        best_val_threshold = float(best_val_threshold_for_checkpoint)
        torch.save(model.state_dict(), fold_dir / "classifier_best.pt")
        torch.save(policy_agent.model.state_dict(), fold_dir / "dqn_best.pt")
        # Backward-compatible filenames now point to the best checkpoint, not the last one.
        torch.save(model.state_dict(), fold_dir / "classifier_final.pt")
        torch.save(policy_agent.model.state_dict(), fold_dir / "dqn_final.pt")
        save_json(
            {
                "best_cycle": int(best_cycle),
                "best_val_score": float(best_val_score),
                "best_checkpoint_metric": best_checkpoint_metric,
                "best_val_threshold": float(best_val_threshold),
                "best_labeled_count": int(best_labeled_count),
                "final_labeled_count": int(len(labeled_idx)),
                "final_queried_count": int(total_queried),
                "note": "classifier_final.pt and dqn_final.pt are aliases of the best validation checkpoint; *_last.pt keeps the last AL-cycle state.",
            },
            fold_dir / "best_checkpoint.json",
        )

        split_metrics: dict[str, dict[str, float]] = {}
        split_outputs: dict[str, pd.DataFrame] = {}
        for split_name, split_indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
            m, out = save_split_outputs(
                model,
                X_all,
                y_all,
                split_indices,
                df,
                split_name,
                fold_dir,
                args,
                self.device,
                threshold=best_val_threshold,
            )
            split_metrics[split_name] = m
            split_outputs[split_name] = out

        if getattr(args, "save_test_heatmaps", False):
            save_full_test_heatmaps(
                model=model,
                X_all=X_all,
                y_all=y_all,
                test_indices=test_indices,
                test_predictions=split_outputs["test"],
                fold_dir=fold_dir,
                args=args,
                device=self.device,
            )

        result = ActiveLearningResult(
            history=history,
            selected_rows=selected_rows,
            best_val_threshold=best_val_threshold,
            labeled_count=int(best_labeled_count),
            best_cycle=int(best_cycle),
            best_val_score=float(best_val_score),
            final_labeled_count=int(len(labeled_idx)),
            final_queried_count=int(total_queried),
        )
        return model, result, split_metrics, split_outputs["test"]


# -----------------------------
# Splitting and CV
# -----------------------------


def subject_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Sort subjects so fold construction is reproducible even if the manifest row
    # order differs across machines. Unlabeled rows (label=-1 / is_labeled=False)
    # are excluded because they must not define train/val/test folds.
    work = df.loc[labeled_mask_from_df(df)].copy()
    return (
        work.groupby("subject_key", as_index=False)
          .agg(label=("label", "max"))
          .sort_values("subject_key")
          .reset_index(drop=True)
    )


def split_train_val_subjects(trainval_subjects: np.ndarray, subject_labels: dict[str, int], val_frac: float, seed: int):
    labels = np.asarray([subject_labels[s] for s in trainval_subjects], dtype=int)
    n_val = max(1, int(round(len(trainval_subjects) * val_frac)))
    if len(trainval_subjects) - n_val < 1:
        raise ValueError("Not enough subjects for train/val split.")
    try:
        if len(np.unique(labels)) == 2 and min(np.bincount(labels, minlength=2)) >= 2 and n_val >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed)
            tr, va = next(sss.split(trainval_subjects, labels))
            return trainval_subjects[tr], trainval_subjects[va]
    except Exception:
        pass
    tr, va = train_test_split(trainval_subjects, test_size=n_val, random_state=seed, shuffle=True)
    return np.asarray(tr), np.asarray(va)


def single_split_subjects(df: pd.DataFrame, seed: int, train_frac: float, val_frac: float):
    sub = subject_frame(df)
    subjects = sub["subject_key"].to_numpy()
    labels = sub["label"].astype(int).to_numpy()
    stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels, minlength=2)) >= 2 else None
    train_subj, temp_subj, _, temp_y = train_test_split(subjects, labels, train_size=train_frac, random_state=seed, stratify=stratify)
    rel_val = val_frac / max(1e-8, 1.0 - train_frac)
    temp_labels = sub.set_index("subject_key").loc[temp_subj, "label"].astype(int).to_numpy()
    stratify_temp = temp_labels if len(np.unique(temp_labels)) == 2 and min(np.bincount(temp_labels, minlength=2)) >= 2 else None
    val_subj, test_subj = train_test_split(temp_subj, train_size=rel_val, random_state=seed, stratify=stratify_temp)
    return train_subj, val_subj, test_subj


def indices_for_subjects(df: pd.DataFrame, subjects: Iterable[str]) -> np.ndarray:
    s = set(map(str, subjects))
    return df.index[df["subject_key"].astype(str).isin(s)].to_numpy(dtype=int)


def summarize_split(df: pd.DataFrame, indices: np.ndarray) -> dict[str, Any]:
    part = df.iloc[np.asarray(indices, dtype=int)] if len(indices) else df.iloc[[]]
    labeled_mask = labeled_mask_from_df(part) if len(part) else np.asarray([], dtype=bool)
    out = {
        "n_images": int(len(part)),
        "n_subjects": int(part["subject_key"].nunique()) if len(part) else 0,
        "n_labeled_images": int(labeled_mask.sum()) if len(part) else 0,
        "n_unlabeled_images": int((~labeled_mask).sum()) if len(part) else 0,
        "label_counts_images": {str(k): int(v) for k, v in part["label"].value_counts().sort_index().to_dict().items()} if len(part) else {},
    }
    labeled_part = part.loc[labeled_mask].copy() if len(part) else part
    if len(labeled_part):
        subj = labeled_part.groupby("subject_key", as_index=False).agg(label=("label", "max"))
        out["label_counts_subjects"] = {str(k): int(v) for k, v in subj["label"].value_counts().sort_index().to_dict().items()}
    else:
        out["label_counts_subjects"] = {}
    if "dataset" in part.columns and len(part):
        out["dataset_counts"] = {str(k): int(v) for k, v in part["dataset"].value_counts().sort_index().to_dict().items()}
    return out


# -----------------------------
# Optional DE optimizer
# -----------------------------



class RandomKeySpace:
    """Grouped random-key search space for classifier backbone + DQN.

    Use --de-group to tune one logical group at a time. This keeps DE from
    searching a huge joint search space and makes results easier to interpret.

    Groups:
      cnn_train  : lr, batch_size, initial_epochs, dropout,
                   weight_decay, loss_name, focal_gamma
      cnn_arch   : base_filters, cnn_stacks, dense_units, activation
      al_train   : retrain_epochs, lr, dropout, weight_decay; evaluated with AL fitness
      dqn_explore: dqn_lr, epsilon_start, epsilon_final, epsilon_decay, label_cost
      dqn_replay : q_hidden_dim, q_mlp_layers, replay_size, replay_batch_size,
                   dqn_updates_per_cycle, dqn_grad_clip
      scope      : scope_alpha, scope_weight, q_policy_temperature, advantage_clip
      joint      : compact 10-param refinement across the most important params
      all        : all grouped params above; available, but not recommended initially
    """

    VALID_GROUPS = ["cnn_train", "cnn_arch", "al_train", "dqn_explore", "dqn_replay", "scope", "joint", "all"]

    def __init__(self, group: str = "cnn_train"):
        self.group = str(group or "cnn_train").lower().replace("-", "_")
        if self.group == "current_dqn_broad":
            self.group = "all"
        if self.group not in self.VALID_GROUPS:
            raise ValueError(f"Unsupported DE group: {group}. Choices: {self.VALID_GROUPS}")

        self.specs: list[tuple[str, str, Any]] = []
        self._build_specs()
        self.dim = int(sum(self._segment_size(spec) for spec in self.specs))

    def _build_specs(self) -> None:
        # Classifier-only fitness can evaluate these parameters directly.
        # retrain_epochs is intentionally excluded because it only has meaning
        # inside the active-learning retraining loop.
        cnn_train = [
            ("lr", "log", (5e-5, 3e-3)),
            ("batch_size", "choice", [8, 16, 32, 64, 96, 128]),
            ("initial_epochs", "int", (30, 120)),
            ("dropout", "linear", (0.0, 0.5)),
            ("weight_decay", "choice", [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]),
            ("loss_name", "choice", ["ce", "focal"]),
            ("focal_gamma", "linear", (0.0, 3.0)),
        ]
        # Active-learning training dynamics. Use --de-fitness-mode al for this group.
        al_train = [
            ("retrain_epochs", "int", (3, 40)),
            ("lr", "log", (5e-5, 3e-3)),
            ("dropout", "linear", (0.0, 0.5)),
            ("weight_decay", "choice", [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]),
        ]
        cnn_arch = [
            ("base_filters", "choice", [8, 16, 24, 32, 48, 64, 96]),
            ("cnn_stacks", "int", (1, 5)),
            ("dense_units", "choice", [32, 64, 128, 256, 512]),
            ("activation", "choice", ["relu", "leaky_relu", "tanh"]),
        ]
        dqn_explore = [
            ("dqn_lr", "log", (1e-5, 5e-3)),
            ("epsilon_start", "linear", (0.2, 1.0)),
            ("epsilon_final", "linear", (0.0, 0.2)),
            ("epsilon_decay", "linear", (0.97, 0.9999)),
            ("label_cost", "linear", (0.05, 1.0)),
        ]
        dqn_replay = [
            ("q_hidden_dim", "choice", [32, 64, 128, 256, 512]),
            ("q_mlp_layers", "int", (1, 5)),
            ("replay_size", "choice", [512, 1024, 2048, 4096, 8192, 16384]),
            ("replay_batch_size", "choice", [16, 32, 64, 128, 256]),
            ("dqn_updates_per_cycle", "choice", [5, 10, 20, 50, 100, 200]),
            ("dqn_grad_clip", "choice", [1.0, 2.0, 5.0, 10.0]),
        ]
        scope = [
            ("scope_alpha", "choice", [0.0, 0.1, 0.2, 0.5, 1.0]),
            ("scope_weight", "choice", [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]),
            ("q_policy_temperature", "log", (0.25, 4.0)),
            ("advantage_clip", "choice", [2.0, 5.0, 10.0]),
        ]
        joint = [
            ("lr", "log", (5e-5, 3e-3)),
            ("retrain_epochs", "int", (3, 40)),
            ("dropout", "linear", (0.0, 0.5)),
            ("weight_decay", "choice", [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]),
            ("base_filters", "choice", [8, 16, 24, 32, 48, 64, 96]),
            ("cnn_stacks", "int", (1, 5)),
            ("dqn_lr", "log", (1e-5, 5e-3)),
            ("epsilon_decay", "linear", (0.97, 0.9999)),
            ("scope_weight", "choice", [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]),
            ("q_policy_temperature", "log", (0.25, 4.0)),
        ]
        if self.group == "cnn_train":
            self.specs = cnn_train
        elif self.group == "cnn_arch":
            self.specs = cnn_arch
        elif self.group == "al_train":
            self.specs = al_train
        elif self.group == "dqn_explore":
            self.specs = dqn_explore
        elif self.group == "dqn_replay":
            self.specs = dqn_replay
        elif self.group == "scope":
            self.specs = scope
        elif self.group == "joint":
            self.specs = joint
        elif self.group == "all":
            self.specs = cnn_train + al_train + cnn_arch + dqn_explore + dqn_replay + scope

    @staticmethod
    def _segment_size(spec: tuple[str, str, Any]) -> int:
        _, kind, values = spec
        return len(values) if kind == "choice" else 1

    def sample(self) -> np.ndarray:
        return np.random.rand(self.dim).astype(np.float32)

    @staticmethod
    def _decode_int(v: float, lo: int, hi: int) -> int:
        return int(np.clip(round(lo + float(v) * (hi - lo)), lo, hi))

    @staticmethod
    def _decode_linear(v: float, lo: float, hi: float) -> float:
        return float(lo + float(v) * (hi - lo))

    @staticmethod
    def _decode_log(v: float, lo: float, hi: float) -> float:
        lo = max(float(lo), 1e-12)
        hi = max(float(hi), lo * 1.0001)
        return float(np.exp(np.log(lo) + float(v) * (np.log(hi) - np.log(lo))))

    def decode(self, key: np.ndarray) -> dict[str, Any]:
        key = np.asarray(key, dtype=np.float32)
        ptr = 0
        out: dict[str, Any] = {}
        for name, kind, values in self.specs:
            if kind == "choice":
                n = len(values)
                idx = int(np.argmax(key[ptr: ptr + n]))
                out[name] = values[idx]
                ptr += n
            elif kind == "int":
                out[name] = self._decode_int(key[ptr], int(values[0]), int(values[1]))
                ptr += 1
            elif kind == "linear":
                out[name] = self._decode_linear(key[ptr], float(values[0]), float(values[1]))
                ptr += 1
            elif kind == "log":
                out[name] = self._decode_log(key[ptr], float(values[0]), float(values[1]))
                ptr += 1
            else:
                raise ValueError(f"Unknown spec type: {kind}")

        # Keep dependent/alias values consistent.
        if "epsilon_start" in out and "epsilon_final" in out:
            out["epsilon_final"] = float(min(float(out["epsilon_final"]), float(out["epsilon_start"])))
        if "replay_batch_size" in out and "replay_size" in out:
            out["replay_batch_size"] = int(min(int(out["replay_batch_size"]), int(out["replay_size"])))
        if "scope_alpha" in out:
            out["scope_gamma"] = float(out["scope_alpha"])
        if "scope_weight" in out:
            out["scope_coef"] = float(out["scope_weight"])
        return out


class DEOptimizer:
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self.rk = RandomKeySpace(getattr(args, "de_group", "cnn_train"))

    def _candidate_args(self, key: np.ndarray) -> tuple[argparse.Namespace, dict[str, Any]]:
        params = self.rk.decode(key)
        tmp = argparse.Namespace(**vars(self.args))
        for k, v in params.items():
            setattr(tmp, k, v)
        tmp = normalize_scope_aliases(tmp)
        return tmp, params

    @staticmethod
    def _score_metrics(metrics: dict[str, float]) -> float:
        auc = metrics.get("auc", float("nan"))
        if not np.isfinite(auc):
            auc = metrics.get("balanced_accuracy", 0.0)
        return float(
            0.35 * metrics.get("f1", 0.0)
            + 0.30 * auc
            + 0.20 * metrics.get("gmean", 0.0)
            + 0.15 * metrics.get("balanced_accuracy", 0.0)
        )

    def _fitness_classifier(
        self,
        tmp: argparse.Namespace,
        params: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        candidate_indices: np.ndarray,
    ) -> float:
        scores: list[float] = []
        eval_splits = self._make_de_eval_splits(df, candidate_indices)
        for split_no, (train_idx, val_idx, split_tag) in enumerate(eval_splits, start=1):
            if len(train_idx) > self.args.de_subset:
                rng = np.random.default_rng(self.args.seed + split_no)
                train_idx = rng.choice(train_idx, size=self.args.de_subset, replace=False)

            model = make_classifier(tmp, self.device)
            fit_classifier(
                model,
                X[train_idx],
                y[train_idx],
                tmp,
                self.device,
                epochs=tmp.initial_epochs,
                verbose=False,
            )

            metrics, _ = evaluate_classifier(
                model,
                X[val_idx],
                y[val_idx],
                tmp.batch_size,
                self.device,
                threshold=0.5,
            )
            score = self._score_metrics(metrics)
            if np.isfinite(score):
                scores.append(float(score))

        return float(np.mean(scores)) if scores else -float("inf")

    def _fitness_active_learning(
        self,
        tmp: argparse.Namespace,
        params: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        candidate_indices: np.ndarray,
    ) -> float:
        """Evaluate a candidate through small classifier-backbone+DQN AL runs.

        DE evaluation can average over one split, 5-fold inner CV, or 5 repeated
        seeds. Candidate indices are expected to be outer train+val only, so the
        outer test fold is never used for hyperparameter selection.
        """
        pool = np.asarray(candidate_indices, dtype=int)
        pool = pool[labeled_mask_from_df(df.iloc[pool])]
        if len(pool) == 0:
            return -float("inf")

        n = min(len(pool), int(self.args.de_subset)) if int(self.args.de_subset) > 0 else len(pool)
        rng = np.random.default_rng(self.args.seed)
        sel = rng.choice(pool, size=n, replace=False) if len(pool) > n else pool.copy()
        sel = np.asarray(sel, dtype=int)

        Xs = X[sel]
        ys = y[sel]
        df_sub = df.iloc[sel].reset_index(drop=True)

        tmp = normalize_scope_aliases(tmp)
        tmp.folds = 1
        tmp.run_de = False
        tmp.de_only = False
        tmp.save_test_heatmaps = False
        tmp.save_epoch_metrics = False
        tmp.verbose = False
        tmp.use_val_optimal_threshold = False
        tmp.until_unlabeled_exhausted = False
        tmp.initial_label_fraction = float(self.args.initial_label_fraction)
        tmp.cycles = int(max(1, min(int(getattr(tmp, "cycles", 3)), int(self.args.de_al_cycles))))
        if getattr(self.args, "de_al_budget", 0) > 0:
            tmp.budget_per_cycle = int(self.args.de_al_budget)
        else:
            tmp.budget_per_cycle = int(self.args.budget_per_cycle)
        if getattr(self.args, "de_al_max_annotations", -1) >= 0:
            tmp.max_annotations = int(self.args.de_al_max_annotations)
        tmp.inference_batch_size = int(min(getattr(tmp, "inference_batch_size", 128), 128))

        scores: list[float] = []
        eval_splits = self._make_de_eval_splits(df_sub, np.arange(len(df_sub), dtype=int))
        for split_no, (train_idx, val_idx, split_tag) in enumerate(eval_splits, start=1):
            with tempfile.TemporaryDirectory(prefix=f"de_al_candidate_{split_tag}_") as td:
                runner = ActiveLearningRunner(tmp, self.device)
                _, al_result, split_metrics, _ = runner.run(
                    Xs,
                    ys,
                    np.asarray(train_idx, dtype=int),
                    np.asarray(val_idx, dtype=int),
                    np.asarray(val_idx, dtype=int),
                    df_sub,
                    Path(td) / "fold_00",
                )

            best_cycle_score = -float("inf")
            for row in getattr(al_result, "history", []) or []:
                val_metrics = {k[4:]: v for k, v in row.items() if str(k).startswith("val_")}
                if val_metrics:
                    score = self._score_metrics(val_metrics)
                    if np.isfinite(score):
                        best_cycle_score = max(best_cycle_score, float(score))
            final_score = self._score_metrics(split_metrics["val"])
            score = float(best_cycle_score) if np.isfinite(best_cycle_score) else float(final_score)
            if np.isfinite(score):
                scores.append(score)

        return float(np.mean(scores)) if scores else -float("inf")

    def _fitness(
        self,
        key: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        candidate_indices: np.ndarray,
    ) -> float:
        params = self.rk.decode(key)
        tmp = argparse.Namespace(**vars(self.args))
        for k, v in params.items():
            setattr(tmp, k, v)

        try:
            if getattr(self.args, "de_fitness_mode", "classifier") == "al":
                return self._fitness_active_learning(tmp, params, X, y, df, candidate_indices)
            return self._fitness_classifier(tmp, params, X, y, df, candidate_indices)
        except RuntimeError as e:
            # Bad candidates can hit OOM/invalid shapes in a broad search. Treat them as
            # failed evaluations instead of terminating the whole DE run.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[DE][WARN] candidate failed: {e}")
            return -float("inf")
        except Exception as e:
            print(f"[DE][WARN] candidate failed: {e}")
            return -float("inf")

    def _mutate(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Create mutant vectors for DE.

        Default is the paper-like cluster-guided mutation:
            v_clu = win_g + F * (x_r1 - x_r2)

        Here fitness is maximized, while the paper describes choosing the cluster
        with the lowest average objective value. We implement this literally by
        setting objective = -fitness and selecting the cluster with the lowest
        mean objective, which is equivalent to the highest mean fitness.
        """
        n, d = pop.shape
        trial = np.empty_like(pop)

        if getattr(self.args, "de_mutation", "paper_cluster") == "template_blend":
            elite_idx = int(np.argmax(fitness))
            elite = pop[elite_idx]
            for i in range(n):
                idxs = [j for j in range(n) if j != i]
                if len(idxs) >= 3:
                    a, b, c = pop[random.sample(idxs, 3)]
                else:
                    a = b = c = pop[random.randrange(n)]
                mutant = elite + self.args.de_F * (a - b) + 0.5 * self.args.de_F * (elite - c)
                trial[i] = np.clip(mutant, 0.0, 1.0)
            return trial

        k = random.randint(2, max(2, min(int(math.sqrt(n)) + 1, n)))
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=self.args.seed)
            labels = km.fit_predict(pop)
            objective = -np.asarray(fitness, dtype=np.float32)
            cluster_objectives = []
            for c_idx in range(k):
                idx = np.where(labels == c_idx)[0]
                cluster_objectives.append(float(objective[idx].mean()) if len(idx) else np.inf)
            target_cluster = int(np.argmin(cluster_objectives))
            cluster_indices = np.where(labels == target_cluster)[0]
            if len(cluster_indices) == 0:
                cluster_indices = np.arange(n)
            elite_idx = int(cluster_indices[np.argmax(fitness[cluster_indices])])
        except Exception:
            cluster_indices = np.arange(n)
            elite_idx = int(np.argmax(fitness))

        elite = pop[elite_idx]
        for i in range(n):
            pool = [j for j in cluster_indices.tolist() if j != i]
            if len(pool) < 2:
                pool = [j for j in range(n) if j != i]
            if len(pool) >= 2:
                r1, r2 = pop[random.sample(pool, 2)]
            else:
                r1 = r2 = pop[random.randrange(n)]
            mutant = elite + self.args.de_F * (r1 - r2)
            trial[i] = np.clip(mutant, 0.0, 1.0)
        return trial

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        candidate_indices: np.ndarray,
        out_dir: Path,
    ) -> dict[str, Any]:
        pop = np.stack([self.rk.sample() for _ in range(self.args.de_pop_size)], axis=0)
        fitness = np.full(self.args.de_pop_size, -np.inf, dtype=np.float32)
        history = []

        candidate_indices = np.asarray(candidate_indices, dtype=int)

        for gen in range(1, self.args.de_generations + 1):
            for i in range(self.args.de_pop_size):
                if not np.isfinite(fitness[i]):
                    fitness[i] = self._fitness(pop[i], X, y, df, candidate_indices)

            best_idx = int(np.argmax(fitness))
            print(
                f"[DE] generation {gen}/{self.args.de_generations}, "
                f"best={fitness[best_idx]:.4f}, params={self.rk.decode(pop[best_idx])}"
            )

            mutant = self._mutate(pop, fitness)
            trial = pop.copy()

            for i in range(self.args.de_pop_size):
                j_rand = random.randrange(self.rk.dim)
                mask = np.random.rand(self.rk.dim) < self.args.de_CR
                mask[j_rand] = True
                trial[i, mask] = mutant[i, mask]

                f_trial = self._fitness(trial[i], X, y, df, candidate_indices)
                if f_trial >= fitness[i]:
                    pop[i] = trial[i]
                    fitness[i] = f_trial

            best_idx = int(np.argmax(fitness))
            history.append({
                "generation": gen,
                "best_score": float(fitness[best_idx]),
                **self.rk.decode(pop[best_idx]),
            })
            pd.DataFrame(history).to_csv(out_dir / "de_history.csv", index=False)

        best_idx = int(np.argmax(fitness))
        result = {
            "best_score": float(fitness[best_idx]),
            "best_params": self.rk.decode(pop[best_idx]),
        }
        print(f"[DE] final best={result['best_score']:.4f}, params={result['best_params']}")
        save_json(result, out_dir / "de_best_params.json")
        return result
    
    def _make_de_inner_split(
        self,
        df: pd.DataFrame,
        candidate_indices: np.ndarray,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        candidate_indices = candidate_indices[labeled_mask_from_df(df.iloc[candidate_indices])]
        part = df.iloc[candidate_indices].copy()
        sub = subject_frame(part)

        subjects = sub["subject_key"].to_numpy()
        labels = sub["label"].astype(int).to_numpy()

        if len(subjects) < 4:
            raise RuntimeError("DE inner split needs at least 4 labeled subjects.")

        split_seed = self.args.seed if seed is None else int(seed)
        stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels, minlength=2)) >= 2 else None

        train_subj, val_subj = train_test_split(
            subjects,
            test_size=0.25,
            random_state=split_seed,
            stratify=stratify,
        )

        subj_values = df.iloc[candidate_indices]["subject_key"].astype(str)
        train_idx = candidate_indices[subj_values.isin(set(map(str, train_subj))).to_numpy()]
        val_idx = candidate_indices[subj_values.isin(set(map(str, val_subj))).to_numpy()]

        return train_idx.astype(int), val_idx.astype(int)

    def _make_de_eval_splits(
        self,
        df: pd.DataFrame,
        candidate_indices: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """Return inner DE evaluation splits.

        --de-eval-mode split: one subject-level train/validation split.
        --de-eval-mode kfold: average over up to --de-eval-folds subject folds.
        --de-eval-mode seeds: average over --de-eval-seeds repeated splits.
        """
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        candidate_indices = candidate_indices[labeled_mask_from_df(df.iloc[candidate_indices])]
        mode = str(getattr(self.args, "de_eval_mode", "split")).lower()
        if mode == "seeds":
            n = max(1, int(getattr(self.args, "de_eval_seeds", 5)))
            return [
                (*self._make_de_inner_split(df, candidate_indices, seed=self.args.seed + i), f"seed_{i + 1:02d}")
                for i in range(n)
            ]
        if mode == "kfold":
            part = df.iloc[candidate_indices].copy()
            sub = subject_frame(part)
            subjects = sub["subject_key"].to_numpy()
            labels = sub["label"].astype(int).to_numpy()
            class_min = int(min(np.bincount(labels, minlength=2))) if len(labels) else 0
            n_splits = min(max(2, int(getattr(self.args, "de_eval_folds", 5))), len(subjects), class_min if class_min > 1 else len(subjects))
            if n_splits < 2:
                tr, va = self._make_de_inner_split(df, candidate_indices, seed=self.args.seed)
                return [(tr, va, "split_01")]
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.args.seed)
            out: list[tuple[np.ndarray, np.ndarray, str]] = []
            subj_values = df.iloc[candidate_indices]["subject_key"].astype(str)
            for fold_no, (tr_s, va_s) in enumerate(splitter.split(subjects, labels), start=1):
                train_subj = set(map(str, subjects[tr_s]))
                val_subj = set(map(str, subjects[va_s]))
                train_idx = candidate_indices[subj_values.isin(train_subj).to_numpy()]
                val_idx = candidate_indices[subj_values.isin(val_subj).to_numpy()]
                out.append((train_idx.astype(int), val_idx.astype(int), f"fold_{fold_no:02d}"))
            return out
        tr, va = self._make_de_inner_split(df, candidate_indices, seed=self.args.seed)
        return [(tr, va, "split_01")]


# -----------------------------
# Argparse and main
# -----------------------------


def parse_args():
    p = argparse.ArgumentParser(description="PyTorch DRL + scope loss active learning for OASIS MRI manifests.")
    p.add_argument("--manifest-csv", type=str, required=True, help="Manifest CSV with image_path,label,subject_key columns.")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--cache-npz", type=str, default=None, help="Optional cache for converted MRI inputs. Use separate cache files for 2D and 2.5D runs.")
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Debug: load only N samples, approximately balanced.")
    p.add_argument("--image-size", type=int, nargs=2, default=[128, 128])
    p.add_argument("--input-mode", type=str, default="2.5d", choices=["2d", "2.5d"],
                   help="2.5d uses sagittal+coronal+axial slices as channels; 2d uses one plane with N slices and averages slice predictions per MRI.")
    p.add_argument("--plane", type=str, default="axial", choices=["axial", "coronal", "sagittal"],
                   help="Plane used only by --input-mode 2d.")
    p.add_argument("--n-slices", type=int, default=15,
                   help="Number of 2D slices per MRI used only by --input-mode 2d.")
    p.add_argument("--slice-aggregation", type=str, default="mean_logits", choices=["mean_logits"],
                   help="How 2D slice predictions are aggregated to one MRI-level prediction.")
    p.add_argument("--slices-per-plane", type=int, default=10,
                   help="Number of 2D slices sampled from each anatomical plane. Default 10 gives 30 input channels total.")
    p.add_argument("--slice-fraction-min", type=float, default=0.15,
                   help="Lowest normalized slice location used when --slices-per-plane > 1. Avoids edge slices by default.")
    p.add_argument("--slice-fraction-max", type=float, default=0.85,
                   help="Highest normalized slice location used when --slices-per-plane > 1. Avoids edge slices by default.")
    p.add_argument("--slice-fractions", type=float, nargs=3, default=[0.5, 0.5, 0.5],
                   help="Backward-compatible 3-slice mode only: sagittal/coronal/axial slice fractions when --slices-per-plane 1.")
    p.add_argument("--slice-indices", type=int, nargs=3, default=None,
                   help="Backward-compatible 3-slice mode only: explicit x/y/z slice indices when --slices-per-plane 1.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # CV/split
    p.add_argument("--folds", type=int, default=5, help="Use 5 for subject-level 5-fold CV. Use 1 for one train/val/test split.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--train-frac", type=float, default=0.70, help="Only used when --folds 1.")

    # Classifier
    p.add_argument("--cnn-arch", type=str, default="current", choices=["current", "resnet18", "densenet121", "basic_cnn"],
                   help="Classifier backbone. current is the original compact 2.5D CNN; the other three are baselines adapted from train_all_2d.py.")
    p.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true",
                   help="Use ImageNet pretrained weights for resnet18/densenet121 when available. Enabled by default.")
    p.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false",
                   help="Use random initialization for resnet18/densenet121 instead of ImageNet weights.")
    p.set_defaults(pretrained_backbone=True)
    p.add_argument("--cnn-stacks", type=int, default=3)
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--dense-units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "leaky_relu", "tanh", "sigmoid"])
    p.add_argument("--loss-name", type=str, default="focal", choices=["focal", "ce"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--inference-batch-size", type=int, default=128)
    p.add_argument("--initial-epochs", type=int, default=10)
    p.add_argument("--retrain-epochs", type=int, default=5)
    p.add_argument("--reset-classifier-each-cycle", action="store_true")
    p.add_argument("--use-val-optimal-threshold", action="store_true")
    p.add_argument("--best-checkpoint-metric", choices=["composite", "auc", "f1", "balanced_accuracy", "gmean", "accuracy"], default="composite",
                   help="Validation metric used to select the best AL/DQN checkpoint for final reporting.")
    p.add_argument("--respect-manifest-splits", action=argparse.BooleanOptionalAction, default=True,
                   help="When --folds 1 and the manifest has split=train/val/test/unlabeled_pool, use those splits instead of re-splitting.")
    p.add_argument("--no-save-epoch-metrics", dest="save_epoch_metrics", action="store_false",
                   help="Disable per-epoch train/val metric logging. By default epoch_metrics.csv and epoch curves are saved.")
    p.set_defaults(save_epoch_metrics=True)

    # Heatmap / saliency explanations for final test set
    p.add_argument("--save-test-heatmaps", action="store_true",
                   help="Save explanation overlays for every image in the final test split of each fold during training.")
    p.add_argument("--heatmap-only", action="store_true",
                   help="External heatmap mode: skip training/DE, load existing fold_XX/classifier_best.pt (or --heatmap-checkpoint), and generate test heatmaps.")
    p.add_argument("--heatmap-checkpoint", type=str, default="best",
                   help="Checkpoint for --heatmap-only: best, final, last, or a path relative to each fold directory / absolute path.")
    p.add_argument("--heatmap-method", type=str, default="input-gradient", choices=["input-gradient", "branch-gradcam", "both"],
                   help="input-gradient saves one slice-specific saliency map per input channel; branch-gradcam saves one Grad-CAM per CNN branch/plane; both saves both.")
    p.add_argument("--heatmap-target", type=str, default="pred", choices=["pred", "true", "demented"],
                   help="Class used for the explanation: predicted class, true class, or always class 1/demented.")
    p.add_argument("--heatmap-alpha", type=float, default=0.45)
    p.add_argument("--heatmap-dpi", type=int, default=180)
    p.add_argument("--heatmap-save-npy", action="store_true", help="Also save raw/normalized explanation arrays as .npy files.")
    p.add_argument("--heatmap-save-all-slice-pngs", action="store_true",
                   help="For input-gradient, additionally save one PNG per input channel/slice. This can create many files.")
    p.add_argument("--heatmap-display-fractions", type=float, nargs=3, default=[0.5, 0.5, 0.5],
                   help="For multi-slice input, choose one display slice per plane for summary PNGs by nearest fraction.")
    p.add_argument("--heatmap-display-indices", type=int, nargs=3, default=None,
                   help="For multi-slice input, choose one local slice number per plane for summary PNGs. Overrides --heatmap-display-fractions.")
    # Active learning / DQN
    p.add_argument("--initial-label-fraction", type=float, default=0.10)
    p.add_argument("--cycles", type=int, default=5,
                   help="Fixed number of active-learning cycles. Ignored when --until-unlabeled-exhausted is enabled.")
    p.add_argument("--until-unlabeled-exhausted", action="store_true",
                   help="Paper-like stopping rule: keep active-learning cycles going until the unlabeled pool is empty or --max-annotations is reached.")
    p.add_argument("--max-annotations", type=int, default=-1,
                   help="Maximum number of newly queried labels after the initial labeled set. -1 means no cap.")
    p.add_argument("--budget-per-cycle", type=int, default=32,
                   help="Maximum newly queried labels per active-learning cycle/pass.")
    p.add_argument("--label-cost", type=float, default=0.2)
    p.add_argument(
        "--reward-mode",
        choices=["centroid", "softmax"],
        default="centroid",
        help=(
            "Reward uncertainty source. centroid is paper-like: feature-to-class-centroid entropy; "
            "softmax uses classifier probability entropy as an ablation/fallback."
        ),
    )
    p.add_argument("--q-hidden-dim", type=int, default=128)
    p.add_argument("--q-mlp-layers", type=int, default=4)
    p.add_argument("--dqn-lr", type=float, default=1e-3)
    p.add_argument("--dqn-gamma", type=float, default=0.95)
    p.add_argument("--epsilon-start", type=float, default=0.8)
    p.add_argument("--epsilon-final", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--scope-alpha", type=float, default=None,
                   help="Scope Loss alpha. If omitted, uses --scope-gamma as a backward-compatible alias.")
    p.add_argument("--scope-gamma", type=float, default=0.2,
                   help="Deprecated alias for --scope-alpha; kept for older notebooks/commands.")
    p.add_argument("--scope-weight", type=float, default=None,
                   help="Weight multiplying the scope loss term. If omitted, uses --scope-coef as an alias.")
    p.add_argument("--scope-coef", type=float, default=0.05,
                   help="Deprecated alias for --scope-weight; kept for older notebooks/commands.")
    p.add_argument("--q-policy-temperature", type=float, default=1.0,
                   help="Temperature used to convert Q-values into a policy for Scope Loss.")
    p.add_argument("--advantage-clip", type=float, default=5.0,
                   help="Clip z-scored advantages to [-value, value] before Scope Loss.")
    p.add_argument("--dqn-grad-clip", type=float, default=5.0,
                   help="Gradient norm clipping for the Q-network.")
    p.add_argument("--replay-size", type=int, default=4096)
    p.add_argument("--replay-batch-size", type=int, default=64)
    p.add_argument("--dqn-updates-per-cycle", type=int, default=50)

    # Debug preset
    p.add_argument("--debug-preset", action="store_true", help="Fast smoke-test settings.")

    # DE optimizer
    p.add_argument("--run-de", action="store_true", help="Run optional DE random-key hyperparameter search before CV.")
    p.add_argument("--de-only", action="store_true", help="Run DE search and exit before final CV/training.")
    p.add_argument("--de-group", choices=RandomKeySpace.VALID_GROUPS, default="cnn_train",
                   help="Grouped DE search space for classifier backbone + DQN. Recommended order for --cnn-arch current: cnn_train -> cnn_arch -> al_train -> dqn_explore -> dqn_replay -> scope -> joint. For baseline backbones, cnn_arch is usually skipped.")
    p.add_argument("--de-range-mode", choices=RandomKeySpace.VALID_GROUPS + ["current_dqn_broad"], default=None,
                   help="Deprecated alias for --de-group. current_dqn_broad maps to all.")
    p.add_argument("--de-pop-size", type=int, default=6)
    p.add_argument("--de-generations", type=int, default=3)
    p.add_argument("--de-F", type=float, default=0.6)
    p.add_argument("--de-CR", type=float, default=0.7)
    p.add_argument("--de-subset", type=int, default=300)
    p.add_argument("--de-eval-mode", choices=["split", "kfold", "seeds"], default="split",
                   help="How each DE candidate is scored: one inner split, k-fold average, or average over multiple random seeds.")
    p.add_argument("--de-eval-folds", type=int, default=5, help="Number of inner folds when --de-eval-mode kfold.")
    p.add_argument("--de-eval-seeds", type=int, default=5, help="Number of repeated inner splits when --de-eval-mode seeds.")
    p.add_argument("--de-fitness-mode", choices=["classifier", "al"], default="classifier",
                   help="classifier tunes supervised CNN quickly; al runs a small active-learning loop so DQN parameters matter.")
    p.add_argument("--de-al-cycles", type=int, default=3,
                   help="Maximum active-learning cycles inside each DE candidate when --de-fitness-mode al.")
    p.add_argument("--de-al-budget", type=int, default=32,
                   help="Cap per-cycle query budget inside each DE candidate when --de-fitness-mode al. Use 0 to not cap decoded budget.")
    p.add_argument("--de-al-max-annotations", type=int, default=96,
                   help="Cap total queried labels inside each DE candidate when --de-fitness-mode al. Use -1 for no cap.")
    p.add_argument("--de-mutation", choices=["paper_cluster", "template_blend"], default="paper_cluster",
                   help="DE mutation rule. paper_cluster uses elite-cluster differential mutation; template_blend is an older ablation.")
    p.add_argument("--apply-de-best", action="store_true", help="Override args with best DE params after search.")
    return p.parse_args()


def normalize_scope_aliases(args):
    # Keep older notebooks working while making the paper terminology explicit.
    if getattr(args, "scope_alpha", None) is None:
        args.scope_alpha = float(args.scope_gamma)
    else:
        args.scope_gamma = float(args.scope_alpha)
    if getattr(args, "scope_weight", None) is None:
        args.scope_weight = float(args.scope_coef)
    else:
        args.scope_coef = float(args.scope_weight)
    args.advantage_clip = abs(float(args.advantage_clip))
    args.q_policy_temperature = max(float(args.q_policy_temperature), 1e-6)
    args.dqn_grad_clip = abs(float(args.dqn_grad_clip))
    return args


def apply_presets(args):
    if args.debug_preset:
        args.limit = args.limit or 60
        args.folds = 1
        args.initial_epochs = 1
        args.retrain_epochs = 1
        args.cycles = 1
        args.until_unlabeled_exhausted = False
        args.budget_per_cycle = min(args.budget_per_cycle, 8)
        args.batch_size = min(args.batch_size, 16)
        args.inference_batch_size = min(args.inference_batch_size, 32)
        args.dqn_updates_per_cycle = min(args.dqn_updates_per_cycle, 5)
    return args


def _load_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        print(f"[WARN] Could not read JSON {path}: {e}")
        return {}


def _copy_namespace(ns: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(ns))


def _apply_config_dict(ns: argparse.Namespace, cfg: dict[str, Any], skip: Optional[set[str]] = None) -> argparse.Namespace:
    skip = set(skip or set())
    for k, v in cfg.items():
        if k in skip:
            continue
        setattr(ns, k, v)
    return ns


def _resolve_heatmap_checkpoint(fold_dir: Path, checkpoint_spec: str | Path) -> Path:
    spec = str(checkpoint_spec or "best")
    key = spec.lower()
    mapping = {
        "best": fold_dir / "classifier_best.pt",
        "final": fold_dir / "classifier_final.pt",
        "last": fold_dir / "classifier_last.pt",
    }
    ckpt = mapping.get(key, Path(spec))
    if not ckpt.is_absolute() and key not in mapping:
        ckpt = fold_dir / ckpt
    return ckpt


def _make_fold_iter_for_heatmaps(args, df: pd.DataFrame, y: np.ndarray, labeled_mask_all: np.ndarray):
    """Reconstruct the same train/val/test folds used by main training."""
    external_unlabeled_idx_all = np.where(~labeled_mask_all)[0].astype(int)
    sub = subject_frame(df)
    subjects = sub["subject_key"].to_numpy()
    labels = sub["label"].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Need both classes to reconstruct classification/CV folds.")

    if args.folds and args.folds > 1:
        if min(np.bincount(labels, minlength=2)) < args.folds:
            raise RuntimeError(
                f"Cannot run {args.folds}-fold heatmaps: smallest subject class has "
                f"{int(min(np.bincount(labels, minlength=2)))} subjects."
            )
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_iter = []
        label_dict = dict(zip(subjects, labels.astype(int)))
        for fold_idx, (trainval_i, test_i) in enumerate(skf.split(subjects, labels), start=1):
            trainval_subj = subjects[trainval_i]
            test_subj = subjects[test_i]
            train_subj, val_subj = split_train_val_subjects(trainval_subj, label_dict, args.val_frac, seed=args.seed + fold_idx)
            fold_iter.append((fold_idx, train_subj, val_subj, test_subj, external_unlabeled_idx_all))
        return fold_iter

    split_col = df["split"].astype(str).str.lower() if "split" in df.columns else pd.Series("", index=df.index)
    has_manifest_splits = {"train", "val", "test"}.issubset(set(split_col.unique()))
    if args.respect_manifest_splits and has_manifest_splits:
        train_idx_fixed = df.index[(split_col == "train") & labeled_mask_all].to_numpy(dtype=int)
        val_idx_fixed = df.index[(split_col == "val") & labeled_mask_all].to_numpy(dtype=int)
        test_idx_fixed = df.index[(split_col == "test") & labeled_mask_all].to_numpy(dtype=int)
        extra_unlabeled_idx = df.index[(split_col == "unlabeled_pool") | (~labeled_mask_all)].to_numpy(dtype=int)
        return [(1, train_idx_fixed, val_idx_fixed, test_idx_fixed, extra_unlabeled_idx)]

    train_subj, val_subj, test_subj = single_split_subjects(df.loc[labeled_mask_all].copy(), args.seed, args.train_frac, args.val_frac)
    return [(1, train_subj, val_subj, test_subj, external_unlabeled_idx_all)]


def _prediction_frame_for_heatmaps(
    model: nn.Module,
    X_all: np.ndarray,
    y_all: np.ndarray,
    test_indices: np.ndarray,
    df: pd.DataFrame,
    fold_dir: Path,
    args,
    device: torch.device,
) -> pd.DataFrame:
    """Load existing test predictions, or recompute them when absent."""
    candidates = [
        fold_dir / "test_eval" / "metrics" / "test_predictions.csv",
        fold_dir / "test_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            pred = pd.read_csv(p)
            required = {"y_true", "y_pred", "prob_demented"}
            if required.issubset(pred.columns):
                return pred.reset_index(drop=True)
            print(f"[WARN] Existing prediction file lacks {sorted(required)}: {p}; recomputing.")
            break

    ckpt_info = _load_json_if_exists(fold_dir / "best_checkpoint.json")
    threshold = float(ckpt_info.get("best_val_threshold", 0.5))
    metrics, pred_df = evaluate_classifier(
        model,
        X_all[test_indices],
        y_all[test_indices],
        int(getattr(args, "batch_size", 32)),
        device,
        threshold=threshold,
    )
    meta = df.iloc[test_indices].reset_index(drop=True)
    out = pd.concat([meta, pred_df.reset_index(drop=True)], axis=1)
    out["prediction_name"] = ["demented" if int(p) == 1 else "non_demented" for p in out["y_pred"]]
    out["eval_split"] = "test"
    metrics_dir = fold_dir / "test_eval" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(metrics_dir / "test_predictions.csv", index=False)
    save_json(metrics, metrics_dir / "test_metrics_from_heatmap_only.json")
    return out


def generate_heatmaps_only(args, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, device: torch.device, out_dir: Path) -> None:
    """External heatmap entry point: load saved best/final/last classifier and skip training."""
    labeled_mask_all = labeled_mask_from_df(df)
    fold_iter = _make_fold_iter_for_heatmaps(args, df, y, labeled_mask_all)
    print(f"[heatmap-only] Generating heatmaps from saved checkpoints in: {out_dir}")

    for fold_item in fold_iter:
        fold_idx, train_obj, val_obj, test_obj, extra_unlabeled_idx = fold_item
        fold_dir = out_dir / f"fold_{fold_idx:02d}"
        if np.asarray(train_obj).dtype.kind in {"i", "u"}:
            test_idx = np.asarray(test_obj, dtype=int)
        else:
            test_idx = indices_for_subjects(df, test_obj)
        test_idx = test_idx[labeled_mask_all[test_idx]]

        if not fold_dir.exists():
            print(f"[WARN] Fold directory not found, skip heatmaps: {fold_dir}")
            continue

        model_args = _copy_namespace(args)
        # If DE/apply-best changed model-shape parameters for this fold, use the
        # fold-specific config to instantiate a model compatible with the saved checkpoint.
        fold_cfg = _load_json_if_exists(fold_dir / "config_after_de.json")
        if fold_cfg:
            preserve_heatmap = {
                "heatmap_only", "save_test_heatmaps", "heatmap_checkpoint",
                "heatmap_method", "heatmap_target", "heatmap_alpha", "heatmap_dpi",
                "heatmap_save_npy", "heatmap_save_all_slice_pngs",
                "heatmap_display_fractions", "heatmap_display_indices",
                "device", "output_dir", "manifest_csv", "cache_npz", "overwrite_cache",
            }
            model_args = _apply_config_dict(model_args, fold_cfg, skip=preserve_heatmap)

        ckpt_path = _resolve_heatmap_checkpoint(fold_dir, getattr(args, "heatmap_checkpoint", "best"))
        if not ckpt_path.exists():
            print(f"[WARN] Checkpoint not found, skip fold {fold_idx}: {ckpt_path}")
            continue

        model = make_classifier(model_args, device)
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        load_state_dict_cpu_safe(model, state, device)
        model.eval()

        test_predictions = _prediction_frame_for_heatmaps(
            model=model,
            X_all=X,
            y_all=y,
            test_indices=test_idx,
            df=df,
            fold_dir=fold_dir,
            args=model_args,
            device=device,
        )
        print(f"[heatmap-only] fold {fold_idx}: checkpoint={ckpt_path.name}, test_cases={len(test_idx)}")
        save_full_test_heatmaps(
            model=model,
            X_all=X,
            y_all=y,
            test_indices=test_idx,
            test_predictions=test_predictions,
            fold_dir=fold_dir,
            args=model_args,
            device=device,
        )



def main():
    args = parse_args()
    # In heatmap-only mode, prefer the training run's config.json so the model
    # architecture, slice settings, folds, and DE-applied hyperparameters match
    # the saved checkpoints. User-supplied heatmap/device/path options are kept.
    if getattr(args, "heatmap_only", False):
        saved_cfg = _load_json_if_exists(Path(args.output_dir) / "config.json")
        if saved_cfg:
            preserve = {
                "heatmap_only", "save_test_heatmaps", "heatmap_checkpoint",
                "heatmap_method", "heatmap_target", "heatmap_alpha", "heatmap_dpi",
                "heatmap_save_npy", "heatmap_save_all_slice_pngs",
                "heatmap_display_fractions", "heatmap_display_indices",
                "device", "output_dir", "manifest_csv", "cache_npz", "overwrite_cache",
            }
            args = _apply_config_dict(args, saved_cfg, skip=preserve)
            print(f"[heatmap-only] Loaded saved training config from {Path(args.output_dir) / 'config.json'}")
    if getattr(args, "de_range_mode", None):
        args.de_group = "all" if args.de_range_mode == "current_dqn_broad" else args.de_range_mode
    args = normalize_scope_aliases(apply_presets(args))
    args.slice_fractions = tuple(float(v) for v in args.slice_fractions)
    if args.slice_indices is not None:
        args.slice_indices = tuple(int(v) for v in args.slice_indices)
    args.heatmap_display_fractions = tuple(float(v) for v in args.heatmap_display_fractions)
    if args.heatmap_display_indices is not None:
        args.heatmap_display_indices = tuple(int(v) for v in args.heatmap_display_indices)
    args.input_mode = str(args.input_mode).lower()
    args.slices_per_plane = max(1, int(args.slices_per_plane))
    args.n_slices = max(1, int(args.n_slices))
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "heatmap_only", False):
        # Do not overwrite the original training config.json; heatmap-only uses it
        # to reconstruct the model/splits compatible with saved checkpoints.
        save_json(vars(args), out_dir / "heatmap_only_config.json")
    else:
        save_json(vars(args), out_dir / "config.json")

    image_size = tuple(args.image_size)
    df, X, y = load_manifest_arrays(
        args.manifest_csv,
        image_size=image_size,
        cache_npz=args.cache_npz,
        limit=args.limit,
        overwrite_cache=args.overwrite_cache,
        slice_fractions=args.slice_fractions,
        slice_indices=args.slice_indices,
        slices_per_plane=args.slices_per_plane,
        slice_fraction_min=args.slice_fraction_min,
        slice_fraction_max=args.slice_fraction_max,
        input_mode=args.input_mode,
        plane=args.plane,
        n_slices=args.n_slices,
    )
    df.to_csv(out_dir / "used_manifest.csv", index=False)
    labeled_mask_all = labeled_mask_from_df(df)
    external_unlabeled_idx_all = np.where(~labeled_mask_all)[0].astype(int)
    labeled_counts = np.bincount(y[labeled_mask_all].astype(int), minlength=2).tolist() if labeled_mask_all.any() else [0, 0]
    print(
        f"Loaded: input_mode={args.input_mode}, X={X.shape}, labeled_y={labeled_counts}, "
        f"unlabeled_pool={len(external_unlabeled_idx_all)}, subjects={df['subject_key'].nunique()}"
    )

    if getattr(args, "heatmap_only", False):
        generate_heatmaps_only(args, df, X, y, device, out_dir)
        return

    sub = subject_frame(df)
    subjects = sub["subject_key"].to_numpy()
    labels = sub["label"].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Need both classes for classification/CV.")

    fold_metrics: list[dict[str, Any]] = []
    runner = ActiveLearningRunner(args, device)
    if args.folds and args.folds > 1:
        if min(np.bincount(labels, minlength=2)) < args.folds:
            raise RuntimeError(
                f"Cannot run {args.folds}-fold CV: smallest subject class has "
                f"{int(min(np.bincount(labels, minlength=2)))} subjects."
            )
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_iter = []
        label_dict = dict(zip(subjects, labels.astype(int)))
        for fold_idx, (trainval_i, test_i) in enumerate(skf.split(subjects, labels), start=1):
            trainval_subj = subjects[trainval_i]
            test_subj = subjects[test_i]
            train_subj, val_subj = split_train_val_subjects(trainval_subj, label_dict, args.val_frac, seed=args.seed + fold_idx)
            fold_iter.append((fold_idx, train_subj, val_subj, test_subj, external_unlabeled_idx_all))
    else:
        split_col = df["split"].astype(str).str.lower() if "split" in df.columns else pd.Series("", index=df.index)
        has_manifest_splits = {"train", "val", "test"}.issubset(set(split_col.unique()))
        if args.respect_manifest_splits and has_manifest_splits:
            train_idx_fixed = df.index[(split_col == "train") & labeled_mask_all].to_numpy(dtype=int)
            val_idx_fixed = df.index[(split_col == "val") & labeled_mask_all].to_numpy(dtype=int)
            test_idx_fixed = df.index[(split_col == "test") & labeled_mask_all].to_numpy(dtype=int)
            extra_unlabeled_idx = df.index[(split_col == "unlabeled_pool") | (~labeled_mask_all)].to_numpy(dtype=int)
            fold_iter = [(1, train_idx_fixed, val_idx_fixed, test_idx_fixed, extra_unlabeled_idx)]
        else:
            train_subj, val_subj, test_subj = single_split_subjects(df.loc[labeled_mask_all].copy(), args.seed, args.train_frac, args.val_frac)
            fold_iter = [(1, train_subj, val_subj, test_subj, external_unlabeled_idx_all)]

    for fold_item in fold_iter:
        fold_idx, train_obj, val_obj, test_obj, extra_unlabeled_idx = fold_item
        print(f"\n================ Fold {fold_idx}/{len(fold_iter)} ================")
        fold_dir = out_dir / f"fold_{fold_idx:02d}"
        if np.asarray(train_obj).dtype.kind in {"i", "u"}:
            train_idx = np.asarray(train_obj, dtype=int)
            val_idx = np.asarray(val_obj, dtype=int)
            test_idx = np.asarray(test_obj, dtype=int)
        else:
            train_idx = indices_for_subjects(df, train_obj)
            val_idx = indices_for_subjects(df, val_obj)
            test_idx = indices_for_subjects(df, test_obj)
        # Supervised splits must contain labels only; external label=-1 rows are
        # passed separately to the AL unlabeled pool.
        train_idx = train_idx[labeled_mask_all[train_idx]]
        val_idx = val_idx[labeled_mask_all[val_idx]]
        test_idx = test_idx[labeled_mask_all[test_idx]]
        extra_unlabeled_idx = np.asarray(extra_unlabeled_idx, dtype=int)

        fold_args = argparse.Namespace(**vars(args))

        if fold_args.run_de:
            de_dir = fold_dir / "de_search"
            de_dir.mkdir(parents=True, exist_ok=True)

            tune_idx = np.concatenate([train_idx, val_idx])

            de = DEOptimizer(fold_args, device)
            best = de.optimize(
                X=X,
                y=y,
                df=df,
                candidate_indices=tune_idx,
                out_dir=de_dir,
            )

            if fold_args.apply_de_best:
                for k, v in best["best_params"].items():
                    setattr(fold_args, k, v)
                save_json(vars(fold_args), fold_dir / "config_after_de.json")

            if getattr(fold_args, "de_only", False):
                print(f"DE-only mode: fold {fold_idx} best params saved to {de_dir / 'de_best_params.json'}")
                continue


        split_summary = {
            "train": summarize_split(df, train_idx),
            "val": summarize_split(df, val_idx),
            "test": summarize_split(df, test_idx),
            "unlabeled_pool": summarize_split(df, extra_unlabeled_idx),
        }
        manifest_dir = fold_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        df.iloc[train_idx].assign(split="train").to_csv(manifest_dir / "train_manifest.csv", index=False)
        df.iloc[val_idx].assign(split="val").to_csv(manifest_dir / "val_manifest.csv", index=False)
        df.iloc[test_idx].assign(split="test").to_csv(manifest_dir / "test_manifest.csv", index=False)
        df.iloc[extra_unlabeled_idx].assign(split="unlabeled_pool").to_csv(manifest_dir / "unlabeled_pool_manifest.csv", index=False)
        save_json(split_summary, fold_dir / "split_summary.json")
        save_json(split_summary, manifest_dir / "split_summary.json")
        plot_split_label_counts(split_summary, fold_dir / "figures")
        print(json.dumps(split_summary, indent=2, ensure_ascii=False))
        runner = ActiveLearningRunner(fold_args, device)
        _, al_result, split_metrics, _ = runner.run(
            X, y, train_idx, val_idx, test_idx, df, fold_dir, extra_unlabeled_indices=extra_unlabeled_idx
        )
        test_metrics = split_metrics["test"]
        fold_metrics.append(
            {
                "fold": fold_idx,
                "metrics": test_metrics,
                "split_metrics": split_metrics,
                "labeled_count": al_result.labeled_count,
                "best_val_threshold": al_result.best_val_threshold,
                "best_cycle": al_result.best_cycle,
                "best_val_score": al_result.best_val_score,
                "final_labeled_count": al_result.final_labeled_count,
                "final_queried_count": al_result.final_queried_count,
                "split_summary": split_summary,
            }
        )
        print(f"  Test metrics: {json.dumps(test_metrics, indent=2)}")

    rows = []
    split_rows = []
    for item in fold_metrics:
        row = {
            "fold": item["fold"],
            "labeled_count": item["labeled_count"],
            "best_val_threshold": item["best_val_threshold"],
            "best_cycle": item.get("best_cycle", 0),
            "best_val_score": item.get("best_val_score", float("nan")),
            "final_labeled_count": item.get("final_labeled_count", item["labeled_count"]),
            "final_queried_count": item.get("final_queried_count", 0),
        }
        row.update(item["metrics"])
        rows.append(row)
        for split_name, metrics in item.get("split_metrics", {}).items():
            split_row = {
                "fold": item["fold"],
                "eval_split": split_name,
                "labeled_count": item["labeled_count"],
                "best_val_threshold": item["best_val_threshold"],
                "best_cycle": item.get("best_cycle", 0),
                "best_val_score": item.get("best_val_score", float("nan")),
                "n_eval_images": item["split_summary"][split_name]["n_images"],
                "n_eval_subjects": item["split_summary"][split_name]["n_subjects"],
            }
            split_row.update(metrics)
            split_rows.append(split_row)
    fold_df = pd.DataFrame(rows)
    split_df = pd.DataFrame(split_rows)
    metrics_dir = out_dir / "cv_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    split_df.to_csv(metrics_dir / "fold_metrics_by_split.csv", index=False)

    summary_rows = []
    for col in ["accuracy", "balanced_accuracy", "auc", "f1", "gmean", "sensitivity", "specificity", "precision", "recall"]:
        if col in fold_df.columns:
            summary_rows.append(
                {
                    "eval_split": "test",
                    "metric": col,
                    "mean": float(fold_df[col].mean()),
                    "std": float(fold_df[col].std(ddof=1)) if len(fold_df) > 1 else 0.0,
                    "min": float(fold_df[col].min()),
                    "max": float(fold_df[col].max()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(metrics_dir / "cv_summary.csv", index=False)

    split_summary_rows = []
    metric_cols = ["accuracy", "balanced_accuracy", "auc", "f1", "gmean", "sensitivity", "specificity", "precision", "recall"]
    if not split_df.empty:
        for split_name, g in split_df.groupby("eval_split"):
            for col in metric_cols:
                if col in g.columns:
                    split_summary_rows.append(
                        {
                            "eval_split": split_name,
                            "metric": col,
                            "mean": float(g[col].mean()),
                            "std": float(g[col].std(ddof=1)) if len(g) > 1 else 0.0,
                            "min": float(g[col].min()),
                            "max": float(g[col].max()),
                        }
                    )
    split_summary_df = pd.DataFrame(split_summary_rows)
    split_summary_df.to_csv(metrics_dir / "cv_summary_by_split.csv", index=False)
    save_json({"folds": fold_metrics, "summary": summary_df.to_dict(orient="records"), "summary_by_split": split_summary_rows}, metrics_dir / "cv_summary.json")
    print("\n================ CV fold metrics ================")
    print(fold_df.to_string(index=False))
    print("\n================ CV summary ================")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
