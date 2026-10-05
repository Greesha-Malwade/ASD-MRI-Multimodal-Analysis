#!/usr/bin/env python3
"""
src/extract_157_structural_embeddings.py

Stage 11D — Structural 512-D MRI Embedding Extraction for the 157 Verified Subjects.

Pipeline:
1. Loads 157 target subjects from results/gat_embeddings/functional_embeddings.csv
   and data/structural/person2_structural_download_manifest.csv.
2. Preprocesses raw T1 NIfTI volumes from data/raw/abide2_t1/ into (128, 128, 128)
   z-score normalized brain voxel arrays saved to data/preprocessed/structural/.
3. Passes preprocessed volumes through Person 1's trained 3D ResNet-18 model
   (models/checkpoints/best_resnet3d.pth) to extract 512-D feature embeddings.
4. Saves newly generated complete structural embeddings as:
   - results/structural_embeddings_157.csv
   - results/structural_embeddings_157_manifest.csv
5. Generates embedding statistics report & PCA visualization check saved to:
   - results/structural_embeddings_157_pca.png
6. Strictly preserves existing artifacts:
   - results/structural_embeddings.csv (original 5-subject baseline preserved)
   - results/gat_embeddings/functional_embeddings.csv (preserved)
   - data/phenotypic/person2_severity_fmri_cohort.csv (preserved)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models.resnet3d import resnet18_3d
from extract_structural_embeddings import extract_resnet3d_512d

RAW_T1_DIR = PROJECT_ROOT / "data" / "raw" / "abide2_t1"
PREPROC_DIR = PROJECT_ROOT / "data" / "preprocessed" / "structural"
CKPT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_resnet3d.pth"

FUNC_CSV_PATH = PROJECT_ROOT / "results" / "gat_embeddings" / "functional_embeddings.csv"
FMRI_COHORT_PATH = PROJECT_ROOT / "data" / "phenotypic" / "person2_severity_fmri_cohort.csv"

OUT_STRUCT_CSV = PROJECT_ROOT / "results" / "structural_embeddings_157.csv"
OUT_MANIFEST_CSV = PROJECT_ROOT / "results" / "structural_embeddings_157_manifest.csv"
OUT_PCA_PNG = PROJECT_ROOT / "results" / "structural_embeddings_157_pca.png"

LOG_FILE = PROJECT_ROOT / "logs" / "extract_157_structural_embeddings.log"


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def preprocess_nifti_to_128cube(nii_path: Path) -> np.ndarray:
    """
    Preprocesses raw T1 NIfTI image into normalized (128, 128, 128) volume array.
    """
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)

    # Convert to PyTorch Tensor for 3D trilinear interpolation to (128, 128, 128)
    tensor_5d = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    resampled_5d = F.interpolate(tensor_5d, size=(128, 128, 128), mode="trilinear", align_corners=False)
    arr = resampled_5d.squeeze().numpy().astype(np.float32)

    # Z-score normalize brain voxels (intensity > 1% of max)
    brain_mask = arr > (0.01 * arr.max())
    if brain_mask.sum() > 0:
        mean_val = arr[brain_mask].mean()
        std_val = arr[brain_mask].std() + 1e-8
        arr_norm = np.zeros_like(arr, dtype=np.float32)
        arr_norm[brain_mask] = (arr[brain_mask] - mean_val) / std_val
    else:
        arr_norm = arr

    return arr_norm


def main():
    setup_logging()
    logging.info("=" * 80)
    logging.info("STEP 11D: STRUCTURAL 512-D MRI EMBEDDING EXTRACTION (157 SUBJECTS)")
    logging.info("=" * 80)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"3D ResNet checkpoint missing: {CKPT_PATH}")

    if not FUNC_CSV_PATH.exists():
        raise FileNotFoundError(f"Functional embeddings missing: {FUNC_CSV_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using compute device: {device}")

    # Load 3D ResNet Model Checkpoint
    model = resnet18_3d(num_classes=2, in_channels=1).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    model.eval()
    logging.info(f"Loaded 3D ResNet checkpoint from {CKPT_PATH}")

    # Read Target Functional Subjects & Phenotypic Metadata
    df_func = pd.read_csv(FUNC_CSV_PATH)
    df_cohort = pd.read_csv(FMRI_COHORT_PATH)

    df_merged = df_func.merge(df_cohort[["participant_id", "site_code"]], on="participant_id", how="left")
    logging.info(f"Loaded target cohort: {len(df_merged)} subjects")

    PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_STRUCT_CSV.parent.mkdir(parents=True, exist_ok=True)

    records = []
    manifest_records = []

    success_count = 0
    fail_preproc = 0
    fail_embed = 0

    t0 = time.time()

    for idx, row in df_merged.iterrows():
        sub_id = str(row["participant_id"]).strip().replace("sub-", "")
        site_code = str(row["site_code"]).strip().replace("ABIDEII-", "")
        split = str(row["split"]).strip()
        severity = str(row["severity_class"]).strip()

        subject_key = f"{site_code}_{sub_id}"
        raw_nii_path = RAW_T1_DIR / f"{subject_key}_raw.nii.gz"
        npy_path = PREPROC_DIR / f"{subject_key}.npy"

        # Step 1: Preprocess to (128, 128, 128) array if not cached
        if npy_path.exists():
            arr_norm = np.load(npy_path)
        elif raw_nii_path.exists():
            try:
                arr_norm = preprocess_nifti_to_128cube(raw_nii_path)
                np.save(npy_path, arr_norm)
            except Exception as e:
                logging.error(f"[{subject_key}] Preprocessing error: {e}")
                fail_preproc += 1
                continue
        else:
            logging.error(f"[{subject_key}] Missing raw NIfTI: {raw_nii_path}")
            fail_preproc += 1
            continue

        # Step 2: Extract 512-D ResNet-18 Embedding
        try:
            tensor = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                emb_512 = extract_resnet3d_512d(model, tensor).cpu().numpy().flatten()
            
            if np.isnan(emb_512).any() or np.isinf(emb_512).any():
                logging.error(f"[{subject_key}] NaN or Inf detected in extracted embedding!")
                fail_embed += 1
                continue

            rec = {
                "participant_id": int(sub_id),
                "split": split,
                "severity_class": severity,
                "site_code": site_code,
                "subject_key": subject_key
            }
            for d in range(512):
                rec[f"embedding_{d}"] = float(emb_512[d])
            records.append(rec)

            manifest_records.append({
                "participant_id": int(sub_id),
                "site_code": site_code,
                "split": split,
                "severity_class": severity,
                "embedding_dimension": 512,
                "source_t1_path": str(raw_nii_path),
                "preprocessed_t1_path": str(npy_path),
                "checkpoint_path": str(CKPT_PATH),
                "extraction_status": "SUCCESS"
            })
            success_count += 1

            if (idx + 1) % 25 == 0 or (idx + 1) == len(df_merged):
                logging.info(f"Processed {idx + 1}/{len(df_merged)} subjects (Success: {success_count})")

        except Exception as e:
            logging.error(f"[{subject_key}] Embedding extraction error: {e}")
            fail_embed += 1

    # Save Artifacts
    df_struct_157 = pd.DataFrame(records)
    df_struct_157.to_csv(OUT_STRUCT_CSV, index=False)
    logging.info(f"Saved structural embeddings (157 subjects) to: {OUT_STRUCT_CSV}")

    df_manifest_157 = pd.DataFrame(manifest_records)
    df_manifest_157.to_csv(OUT_MANIFEST_CSV, index=False)
    logging.info(f"Saved structural embedding manifest to: {OUT_MANIFEST_CSV}")

    # Generate PCA Plot & Statistical Sanity Check
    emb_cols = [f"embedding_{d}" for d in range(512)]
    X_emb = df_struct_157[emb_cols].values

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_emb)

    plt.figure(figsize=(8, 6))
    for sev, color in [("Low", "blue"), ("Moderate", "green"), ("High", "red")]:
        mask = df_struct_157["severity_class"] == sev
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=sev, alpha=0.7, c=color, s=40)
    plt.title(f"PCA Sanity Check: 512-D Structural Embeddings (157 Subjects)\nEVR: PC1={pca.explained_variance_ratio_[0]:.2f}, PC2={pca.explained_variance_ratio_[1]:.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PCA_PNG, dpi=150)
    plt.close()
    logging.info(f"Saved PCA sanity check plot to: {OUT_PCA_PNG}")

    # Final Summary Logging
    total_time = time.time() - t0
    logging.info("=" * 80)
    logging.info("STEP 11D SUMMARY REPORT")
    logging.info("=" * 80)
    logging.info(f"Target subjects:                  {len(df_merged)}")
    logging.info(f"Successfully preprocessed:        {success_count} / {len(df_merged)}")
    logging.info(f"Successfully embedded:           {success_count} / {len(df_merged)}")
    logging.info(f"Failed preprocessing:            {fail_preproc}")
    logging.info(f"Failed embedding extraction:      {fail_embed}")
    logging.info(f"Structural embedding dimension:   512")
    logging.info(f"NaN count:                        {np.isnan(X_emb).sum()}")
    logging.info(f"Inf count:                        {np.isinf(X_emb).sum()}")
    logging.info(f"All-zero vectors:                 {(X_emb == 0).all(axis=1).sum()}")
    logging.info(f"Unique participant IDs:           {df_struct_157['participant_id'].nunique()}")
    logging.info(f"Total processing time:            {total_time:.2f}s")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
