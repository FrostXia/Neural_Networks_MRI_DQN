from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from oasis_common import build_oasis1_manifest, is_labeled_mask, save_json, split_labeled_unlabeled_subjects


def parse_args():
    parser = argparse.ArgumentParser(description="Build OASIS1 manifests for CNN/DQN active-learning training.")
    parser.add_argument("--oasis1-root", type=str, required=True, help="Root containing OASIS1 RAW discs, e.g. /.../OASIS1/RAW")
    parser.add_argument("--oasis1-clinical", type=str, required=True, help="OASIS1 cross-sectional csv/xlsx")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-oasis1-unlabeled",
        action="store_true",
        help="Do not keep OASIS1 rows with missing/invalid CDR. By default they are saved to unlabeled_pool_manifest.csv.",
    )
    return parser.parse_args()


def summarize(df: pd.DataFrame) -> dict:
    out = {
        "n_images": int(len(df)),
        "n_subjects": int(df["subject_key"].nunique()) if "subject_key" in df else 0,
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()} if len(df) and "label" in df else {},
    }
    if len(df) and "label" in df:
        labeled_mask = is_labeled_mask(df)
        out["n_labeled_images"] = int(labeled_mask.sum())
        out["n_unlabeled_images"] = int((~labeled_mask).sum())
        out["labeled_label_counts"] = {
            str(k): int(v) for k, v in df.loc[labeled_mask, "label"].value_counts().sort_index().to_dict().items()
        }
    if "dataset" in df and len(df):
        out["dataset_counts"] = {str(k): int(v) for k, v in df["dataset"].value_counts().to_dict().items()}
    return out


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    oasis1 = build_oasis1_manifest(
        args.oasis1_root,
        args.oasis1_clinical,
        include_unlabeled=not args.drop_oasis1_unlabeled,
    )
    if oasis1.empty:
        raise RuntimeError("No OASIS1 images were matched. Check --oasis1-root and --oasis1-clinical.")

    oasis1 = oasis1.drop_duplicates(subset=["dataset", "mri_id", "image_path"]).reset_index(drop=True)
    oasis1.to_csv(manifest_dir / "oasis1_manifest.csv", index=False)
    # Keep the combined name for compatibility with existing notebooks/scripts.
    oasis1.to_csv(manifest_dir / "combined_manifest.csv", index=False)

    summary = {"oasis1": summarize(oasis1), "combined": summarize(oasis1)}
    n_unlabeled = int((~is_labeled_mask(oasis1)).sum()) if len(oasis1) else 0
    print(f"OASIS1 matched images: {len(oasis1)} (unlabeled pool candidates: {n_unlabeled})")

    train_df, val_df, test_df, unlabeled_df = split_labeled_unlabeled_subjects(
        oasis1,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    train_df.to_csv(manifest_dir / "train_manifest.csv", index=False)
    val_df.to_csv(manifest_dir / "val_manifest.csv", index=False)
    test_df.to_csv(manifest_dir / "test_manifest.csv", index=False)
    unlabeled_df.to_csv(manifest_dir / "unlabeled_pool_manifest.csv", index=False)

    al_manifest = pd.concat([train_df, val_df, test_df, unlabeled_df], ignore_index=True)
    al_manifest.to_csv(manifest_dir / "al_manifest.csv", index=False)

    split_summary = {
        "train": summarize(train_df),
        "val": summarize(val_df),
        "test": summarize(test_df),
        "unlabeled_pool": summarize(unlabeled_df),
    }
    summary["splits"] = split_summary
    save_json(summary, manifest_dir / "manifest_summary.json")

    print("\nManifest summary")
    print(pd.DataFrame(
        [
            {"split": "combined", **summary["combined"]},
            {"split": "train", **split_summary["train"]},
            {"split": "val", **split_summary["val"]},
            {"split": "test", **split_summary["test"]},
            {"split": "unlabeled_pool", **split_summary["unlabeled_pool"]},
        ]
    ).to_string(index=False))
    print(f"\nSaved OASIS1 manifests to: {manifest_dir}")
    print(f"Use --manifest-csv {manifest_dir / 'al_manifest.csv'} for active-learning runs that include the external unlabeled pool.")


if __name__ == "__main__":
    main()
