from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
    except Exception:
        pass


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_clinical_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported clinical table format: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _canon(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    canon_to_col = {_canon(c): c for c in df.columns}
    for cand in candidates:
        key = _canon(cand)
        if key in canon_to_col:
            return canon_to_col[key]
    for cand in candidates:
        key = _canon(cand)
        for col in df.columns:
            if key and key in _canon(col):
                return col
    if required:
        raise KeyError(f"Could not find any of these columns: {list(candidates)}. Existing columns: {list(df.columns)}")
    return None


def parse_cdr(value) -> float:
    if pd.isna(value):
        return math.nan
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"nan", "na", "n/a", "none", "null"}:
            return math.nan
        value = value.replace(",", ".")
    try:
        return float(value)
    except Exception:
        return math.nan


def label_from_cdr(cdr: float) -> Optional[int]:
    if pd.isna(cdr):
        return None
    if cdr == 0:
        return 0
    if cdr >= 0.5:
        return 1
    return None


def label_name(label: int) -> str:
    if int(label) < 0:
        return "unlabeled"
    return "demented" if int(label) == 1 else "non_demented"


def is_labeled_value(label) -> bool:
    """Return True only for supervised class labels used by train/val/test."""
    try:
        if pd.isna(label):
            return False
        return int(float(label)) in {0, 1}
    except Exception:
        return False


def parse_bool_value(value) -> bool:
    """Parse bool-like CSV values without treating the string "False" as True."""
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


def is_labeled_mask(df: pd.DataFrame) -> pd.Series:
    if "is_labeled" in df.columns:
        return df["is_labeled"].apply(parse_bool_value)
    return df["label"].apply(is_labeled_value)


def extract_mri_id(text: str, dataset: str = "oasis1") -> Optional[str]:
    """Extract an OASIS1 MRI ID such as OAS1_0001_MR1."""
    m = re.search(r"(OAS1_\d+_MR\d+)", text)
    return m.group(1) if m else None


def subject_from_mri_id(mri_id: str) -> str:
    m = re.match(r"(OAS1_\d+)_MR\d+", mri_id)
    if not m:
        raise ValueError(f"Could not parse OASIS1 subject ID from MRI ID: {mri_id}")
    return m.group(1)


def scan_oasis1_images(root: str | Path) -> dict[str, str]:
    root = Path(root)
    image_map: dict[str, str] = {}

    preferred = list(root.rglob("*_111_t88_masked_gfc.hdr"))
    for p in preferred:
        s = str(p)
        if "FSL_SEG" in s or s.endswith("_fseg.hdr"):
            continue
        if "T88_111" not in s:
            continue
        mri_id = extract_mri_id(s, "oasis1")
        if mri_id and mri_id not in image_map:
            image_map[mri_id] = str(p)

    if image_map:
        return image_map

    for p in root.rglob("T1.mgz"):
        mri_id = extract_mri_id(str(p), "oasis1")
        if mri_id and mri_id not in image_map:
            image_map[mri_id] = str(p)
    return image_map


def build_oasis1_manifest(oasis_root: str | Path, clinical_table: str | Path, include_unlabeled: bool = True) -> pd.DataFrame:
    """Build an OASIS1 manifest and keep unlabeled scans for the AL pool.

    Rows with CDR that maps to {0, 1} are marked ``is_labeled=True`` and can be
    used for train/val/test. Rows with missing/invalid CDR are kept as
    ``label=-1`` and ``is_labeled=False`` when ``include_unlabeled=True``; these
    rows are written to the unlabeled-pool manifest instead of being silently
    dropped.
    """
    df = read_clinical_table(clinical_table)
    id_col = find_column(df, ["ID", "Subject ID", "Subject", "MRI ID", "MR ID"])
    cdr_col = find_column(df, ["CDR", "Clinical Dementia Rating"])
    age_col = find_column(df, ["Age"], required=False)
    sex_col = find_column(df, ["M/F", "Sex", "Gender"], required=False)
    mmse_col = find_column(df, ["MMSE"], required=False)

    image_map = scan_oasis1_images(oasis_root)
    rows = []
    for _, row in df.iterrows():
        raw_id = str(row[id_col]).strip()
        mri_id = extract_mri_id(raw_id, "oasis1") or f"{raw_id}_MR1"
        subject_id = subject_from_mri_id(mri_id)
        cdr = parse_cdr(row[cdr_col])
        label = label_from_cdr(cdr)
        is_labeled = label is not None
        if (not is_labeled) and (not include_unlabeled):
            continue
        image_path = image_map.get(mri_id)
        if not image_path:
            continue
        stored_label = int(label) if is_labeled else -1
        rows.append(
            {
                "dataset": "OASIS1",
                "subject_id": subject_id,
                "subject_key": f"OASIS1:{subject_id}",
                "mri_id": mri_id,
                "image_path": image_path,
                "cdr": cdr,
                "label": stored_label,
                "label_name": label_name(stored_label),
                "is_labeled": bool(is_labeled),
                "pool": "labeled" if is_labeled else "unlabeled",
                "age": row[age_col] if age_col else np.nan,
                "sex": row[sex_col] if sex_col else "",
                "mmse": row[mmse_col] if mmse_col else np.nan,
                "image_source": "PROCESSED/T88_111/masked_gfc" if "masked_gfc" in image_path else "fallback",
            }
        )
    return pd.DataFrame(rows)


def split_subjects(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    work = df.copy()
    work = work[is_labeled_mask(work)].copy()
    if work.empty:
        raise ValueError("No labeled rows are available for train/val/test splitting.")

    subject_df = work.groupby("subject_key", as_index=False).agg(label=("label", "max"))
    subjects = subject_df["subject_key"].to_numpy()
    labels = subject_df["label"].astype(int).to_numpy()

    stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels, minlength=2)) >= 2 else None
    train_subj, temp_subj, _, temp_y = train_test_split(
        subjects,
        labels,
        train_size=train_frac,
        random_state=seed,
        stratify=stratify,
    )

    relative_val = val_frac / (val_frac + test_frac)
    temp_labels = subject_df.set_index("subject_key").loc[temp_subj, "label"].astype(int).to_numpy()
    stratify_temp = temp_labels if len(np.unique(temp_labels)) == 2 and min(np.bincount(temp_labels, minlength=2)) >= 2 else None
    val_subj, test_subj = train_test_split(
        temp_subj,
        train_size=relative_val,
        random_state=seed,
        stratify=stratify_temp,
    )

    train_df = work[work["subject_key"].isin(train_subj)].copy()
    val_df = work[work["subject_key"].isin(val_subj)].copy()
    test_df = work[work["subject_key"].isin(test_subj)].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return train_df, val_df, test_df


def split_labeled_unlabeled_subjects(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split labeled rows into train/val/test and keep all unlabeled rows for the AL pool."""
    work = df.copy()
    mask = is_labeled_mask(work)
    labeled = work[mask].copy()
    unlabeled = work[~mask].copy()
    train_df, val_df, test_df = split_subjects(labeled, train_frac, val_frac, test_frac, seed)
    if len(unlabeled):
        unlabeled["split"] = "unlabeled_pool"
        unlabeled["pool"] = "unlabeled"
        unlabeled["is_labeled"] = False
        unlabeled["label"] = -1
        unlabeled["label_name"] = "unlabeled"
    return train_df, val_df, test_df, unlabeled


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = safe_div(tn, tn + fp)
    sensitivity = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    balanced_accuracy = (sensitivity + specificity) / 2.0
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else math.nan
    except Exception:
        auc = math.nan
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "auc": float(auc) if not math.isnan(auc) else math.nan,
        "sensitivity": sensitivity,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def get_monai_transforms(spatial_size: tuple[int, int, int], train: bool = False, pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)):
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        RandAffined,
        RandFlipd,
        RandGaussianNoised,
        ResizeWithPadOrCropd,
        ScaleIntensityd,
        Spacingd,
    )

    transforms = [
        LoadImaged(keys=["image"], reader="NibabelReader", image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        CropForegroundd(keys=["image"], source_key="image"),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size),
    ]
    if train:
        transforms.extend(
            [
                RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
                RandAffined(
                    keys=["image"],
                    prob=0.7,
                    rotate_range=(0.12, 0.12, 0.12),
                    translate_range=(8, 8, 8),
                    scale_range=(0.08, 0.08, 0.08),
                    mode="bilinear",
                    padding_mode="border",
                ),
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.02),
            ]
        )
    transforms.append(EnsureTyped(keys=["image"]))
    return Compose(transforms)


def dataframe_to_monai_records(df: pd.DataFrame, include_metadata: bool = True) -> list[dict]:
    """Convert a manifest DataFrame into MONAI dictionary records.

    For DataLoader training/evaluation, use include_metadata=False. MONAI's
    default list_data_collate tries to stack every key in the dictionary;
    metadata columns such as sex/group/image_source may mix strings and NaNs and
    can crash collation with "must be real number, not str".

    For visualization, include_metadata=True keeps fields such as mri_id/CDR so
    QC PNG titles remain informative.
    """
    records = []
    for _, row in df.iterrows():
        if include_metadata:
            item = row.to_dict()
        else:
            item = {}
        item["image"] = row["image_path"]
        item["label"] = int(row["label"])
        records.append(item)
    return records

def find_optimal_threshold(y_true, y_prob):
    """
    基于真实标签和预测概率，计算 Youden Index 最大时的最佳概率阈值
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    if len(np.unique(y_true)) < 2:
        return 0.5
        
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    return float(optimal_threshold)