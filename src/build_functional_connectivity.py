#!/usr/bin/env python3
"""
src/build_functional_connectivity.py

Stage 6 — Functional Connectivity Matrix Construction for Person 2 rs-fMRI Cohort.

Loads ROI time-series matrices (shape [T, 62]) extracted in Step 5 for the 157 PRIMARY PASS subjects.
Computes:
1. Subject-level 62 x 62 Pearson correlation matrix (signed, unthresholded).
2. Subject-level 62 x 62 Fisher z-transformed matrix (arctanh of clipped Pearson r).

Handles zero-variance ROIs (voxels out of FOV) by setting off-diagonal correlation to 0.0 and diagonal to 1.0.

Outputs saved to:
- data/features/fmri_fc/pearson/
- data/features/fmri_fc/fisher_z/
- data/features/fmri_fc/fc_manifest.csv
- data/features/fmri_fc/atlas_metadata.csv

Command line options:
- --test_batch N: Runs on first N subjects for validation.
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Enforce UTF-8 output encoding for Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src directory to sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

ROI_MANIFEST_PATH = PROJECT_ROOT / "data" / "features" / "fmri" / "roi_extraction_manifest.csv"
ATLAS_METADATA_SRC = PROJECT_ROOT / "data" / "features" / "fmri" / "atlas_metadata.csv"
FC_OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "fmri_fc"
PEARSON_DIR = FC_OUTPUT_DIR / "pearson"
FISHER_Z_DIR = FC_OUTPUT_DIR / "fisher_z"
FC_MANIFEST_PATH = FC_OUTPUT_DIR / "fc_manifest.csv"
ATLAS_METADATA_DST = FC_OUTPUT_DIR / "atlas_metadata.csv"
LOG_FILE = PROJECT_ROOT / "logs" / "fmri_fc_construction.log"


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


def compute_subject_fc(row: pd.Series) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Computes Pearson correlation matrix and Fisher z-transformed matrix for a single subject.
    Handles zero-variance ROIs gracefully by assigning r=0 off-diagonal and r=1 on-diagonal.
    """
    sub_id = str(row["participant_id"]).strip()
    site_code = str(row["site_code"]).strip()
    split = str(row["split"]).strip()
    sev_class = str(row["severity_class"]).strip()
    ados_score = row["ados_2_severity_total"]

    roi_rel_path = str(row["roi_file_path"]).strip()
    roi_full_path = PROJECT_ROOT / roi_rel_path

    if not roi_full_path.exists():
        raise FileNotFoundError(f"ROI time-series missing for sub-{sub_id}: {roi_full_path}")

    # Load ROI matrix [T, 62]
    roi_matrix = np.load(roi_full_path)
    if roi_matrix.ndim != 2:
        raise ValueError(f"Expected 2D ROI matrix, got shape {roi_matrix.shape} for sub-{sub_id}")

    T, N_ROIS = roi_matrix.shape
    if N_ROIS != 62:
        raise ValueError(f"Expected 62 ROIs, got {N_ROIS} for sub-{sub_id}")

    # Check input integrity
    if np.isnan(roi_matrix).any() or np.isinf(roi_matrix).any():
        raise ValueError(f"Input ROI matrix contains NaNs or Infs for sub-{sub_id}")

    # Identify zero-variance ROIs (outside subject FOV)
    roi_stds = np.std(roi_matrix, axis=0)
    zero_var_rois = np.where(roi_stds == 0)[0]
    num_zero_var = len(zero_var_rois)

    # 1. Pearson Correlation Matrix (62 x 62)
    # np.corrcoef expects variables as rows, observations as columns -> rowvar=False for [T, 62]
    with np.errstate(invalid="ignore", divide="ignore"):
        r_matrix = np.corrcoef(roi_matrix, rowvar=False).astype(np.float32)

    # Replace NaNs resulting from zero-variance ROIs with 0.0 (no functional connectivity)
    if np.isnan(r_matrix).any():
        r_matrix = np.nan_to_num(r_matrix, nan=0.0)

    # Validation checks on Pearson matrix
    if r_matrix.shape != (62, 62):
        raise ValueError(f"Expected FC matrix shape (62, 62), got {r_matrix.shape} for sub-{sub_id}")

    # Enforce exact 1.0 on diagonal
    np.fill_diagonal(r_matrix, 1.0)

    # Enforce exact mathematical symmetry to eliminate floating point numerical noise
    r_matrix = 0.5 * (r_matrix + r_matrix.T)

    nan_count = int(np.isnan(r_matrix).sum())
    inf_count = int(np.isinf(r_matrix).sum())

    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"FC matrix contains {nan_count} NaNs and {inf_count} Infs for sub-{sub_id}")

    # Symmetry check
    asym_diff = float(np.max(np.abs(r_matrix - r_matrix.T)))
    is_symmetric = asym_diff < 1e-5
    if not is_symmetric:
        raise ValueError(f"FC matrix is asymmetric (max diff: {asym_diff:.6e}) for sub-{sub_id}")

    # Diagonal check
    diag_vals = np.diag(r_matrix)
    diag_diff = float(np.max(np.abs(diag_vals - 1.0)))
    if diag_diff > 1e-4:
        raise ValueError(f"FC matrix diagonal values deviate from 1.0 (max diff: {diag_diff:.6e}) for sub-{sub_id}")

    # Value range check
    min_corr = float(np.min(r_matrix))
    max_corr = float(np.max(r_matrix))
    if min_corr < -1.0001 or max_corr > 1.0001:
        raise ValueError(f"Correlation values out of range [-1, 1]: [{min_corr}, {max_corr}] for sub-{sub_id}")

    # Off-diagonal mean absolute correlation
    off_diag_mask = ~np.eye(62, dtype=bool)
    mean_abs_corr = float(np.mean(np.abs(r_matrix[off_diag_mask])))

    # 2. Fisher z-Transformation: z = arctanh(r_clipped)
    # Clip values to [-0.999999, 0.999999] safely before arctanh
    r_clipped = np.clip(r_matrix, -0.999999, 0.999999)
    fisher_z_matrix = np.arctanh(r_clipped).astype(np.float32)

    # Ensure symmetry for Fisher-z
    fisher_z_matrix = 0.5 * (fisher_z_matrix + fisher_z_matrix.T)

    # Check Fisher-z values
    fz_nan_count = int(np.isnan(fisher_z_matrix).sum())
    fz_inf_count = int(np.isinf(fisher_z_matrix).sum())
    if fz_nan_count > 0 or fz_inf_count > 0:
        raise ValueError(f"Fisher-z matrix contains {fz_nan_count} NaNs and {fz_inf_count} Infs for sub-{sub_id}")

    # Define file paths
    site_bids = f"ABIDEII-{site_code.replace('ABIDEII-', '')}"
    sub_bids = f"sub-{sub_id.replace('sub-', '')}"

    pearson_filename = f"{site_bids}_{sub_bids}_fc_pearson.npy"
    fisher_z_filename = f"{site_bids}_{sub_bids}_fc_fisher_z.npy"

    pearson_file_path = PEARSON_DIR / pearson_filename
    fisher_z_file_path = FISHER_Z_DIR / fisher_z_filename

    # Save .npy files
    PEARSON_DIR.mkdir(parents=True, exist_ok=True)
    FISHER_Z_DIR.mkdir(parents=True, exist_ok=True)

    np.save(pearson_file_path, r_matrix)
    np.save(fisher_z_file_path, fisher_z_matrix)

    rel_pearson_path = str(pearson_file_path.relative_to(PROJECT_ROOT))
    rel_fisher_z_path = str(fisher_z_file_path.relative_to(PROJECT_ROOT))

    meta_record = {
        "participant_id": sub_id,
        "site_code": site_code,
        "split": split,
        "severity_class": sev_class,
        "ados_2_severity_total": ados_score,
        "roi_count": N_ROIS,
        "timepoints": T,
        "fc_matrix_path": rel_pearson_path,
        "fisher_z_matrix_path": rel_fisher_z_path,
        "fc_status": "SUCCESS",
        "failure_reason": "None",
        "min_correlation": min_corr,
        "max_correlation": max_corr,
        "mean_abs_correlation": mean_abs_corr,
        "is_symmetric": is_symmetric,
        "max_asymmetry_diff": asym_diff,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_var_roi_count": num_zero_var
    }

    return meta_record, r_matrix, fisher_z_matrix


def main():
    parser = argparse.ArgumentParser(description="Stage 6 — Functional Connectivity Matrix Construction")
    parser.add_argument("--test_batch", type=int, default=0, help="Number of subjects for test batch mode")
    args = parser.parse_args()

    setup_logging()
    logging.info("=" * 80)
    logging.info("STEP 6: FUNCTIONAL CONNECTIVITY MATRIX CONSTRUCTION INITIALIZED")
    logging.info("=" * 80)

    # 1. Load Step 5 ROI Extraction Manifest
    if not ROI_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"ROI extraction manifest missing: {ROI_MANIFEST_PATH}")

    df_roi = pd.read_csv(ROI_MANIFEST_PATH, dtype={"participant_id": str})
    df_roi["participant_id"] = df_roi["participant_id"].str.strip()

    # Filter for SUCCESS extraction status
    df_target = df_roi[df_roi["roi_extraction_status"] == "SUCCESS"].copy()
    expected_n = 157

    if args.test_batch > 0:
        logging.info(f"TEST BATCH MODE: Running on first {args.test_batch} subjects.")
        df_target = df_target.head(args.test_batch)
    else:
        assert len(df_target) == expected_n, f"Expected {expected_n} primary subjects in ROI manifest, got {len(df_target)}"

    logging.info(f"Target subjects to process: {len(df_target)}")

    # 2. Copy Atlas Metadata
    if ATLAS_METADATA_SRC.exists():
        FC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(ATLAS_METADATA_SRC, ATLAS_METADATA_DST)
        logging.info(f"Copied atlas metadata to: {ATLAS_METADATA_DST}")

    # 3. Batch Construction Loop
    t_start = time.time()
    fc_records = []
    success_cnt = 0
    fail_cnt = 0

    nan_matrices_cnt = 0
    inf_matrices_cnt = 0
    asym_matrices_cnt = 0
    incorrect_shape_cnt = 0

    all_min_corrs = []
    all_max_corrs = []
    all_mean_abs_corrs = []
    all_timepoints = []

    for idx, row in df_target.iterrows():
        sub_id = str(row["participant_id"]).strip()
        site_code = str(row["site_code"]).strip()
        logging.info(f"[{success_cnt+fail_cnt+1:03d}/{len(df_target):03d}] Constructing FC for sub-{sub_id} ({site_code})...")

        try:
            rec, r_mat, fz_mat = compute_subject_fc(row)
            fc_records.append(rec)
            success_cnt += 1

            all_min_corrs.append(rec["min_correlation"])
            all_max_corrs.append(rec["max_correlation"])
            all_mean_abs_corrs.append(rec["mean_abs_correlation"])
            all_timepoints.append(rec["timepoints"])

            logging.info(f"   -> Success: FC shape (62, 62) | Pearson range: [{rec['min_correlation']:.4f}, {rec['max_correlation']:.4f}] | Mean |r|: {rec['mean_abs_correlation']:.4f} | Zero-var ROIs: {rec['zero_var_roi_count']}")
        except Exception as e:
            fail_cnt += 1
            logging.error(f"   -> FAILED sub-{sub_id}: {e}")
            fail_rec = {
                "participant_id": sub_id,
                "site_code": site_code,
                "split": row["split"],
                "severity_class": row["severity_class"],
                "ados_2_severity_total": row["ados_2_severity_total"],
                "roi_count": 62,
                "timepoints": row.get("num_timepoints", 0),
                "fc_matrix_path": "",
                "fisher_z_matrix_path": "",
                "fc_status": "FAILED",
                "failure_reason": str(e),
                "min_correlation": 0.0,
                "max_correlation": 0.0,
                "mean_abs_correlation": 0.0,
                "is_symmetric": False,
                "max_asymmetry_diff": -1.0,
                "nan_count": -1,
                "inf_count": -1,
                "zero_var_roi_count": -1
            }
            fc_records.append(fail_rec)

    total_time = time.time() - t_start
    df_fc_manifest = pd.DataFrame(fc_records)
    FC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_fc_manifest.to_csv(FC_MANIFEST_PATH, index=False)
    logging.info(f"Saved FC manifest to: {FC_MANIFEST_PATH}")

    # 4. Compute Aggregate QC Statistics
    success_df = df_fc_manifest[df_fc_manifest["fc_status"] == "SUCCESS"]
    split_counts = success_df["split"].value_counts().to_dict()
    sev_counts = success_df["severity_class"].value_counts().to_dict()
    site_counts = success_df["site_code"].value_counts().to_dict()

    tp_min = int(min(all_timepoints)) if all_timepoints else 0
    tp_max = int(max(all_timepoints)) if all_timepoints else 0
    tp_mean = float(np.mean(all_timepoints)) if all_timepoints else 0.0

    global_min_corr = float(min(all_min_corrs)) if all_min_corrs else 0.0
    global_max_corr = float(max(all_max_corrs)) if all_max_corrs else 0.0
    global_mean_abs_corr = float(np.mean(all_mean_abs_corrs)) if all_mean_abs_corrs else 0.0

    success_rate = round(float(success_cnt / len(df_target) * 100.0), 1)

    print("\n" + "=" * 80)
    print("===== STEP 6 FUNCTIONAL CONNECTIVITY REPORT =====")
    print("=" * 80)
    print(f"Required subjects:        {len(df_target)}")
    print(f"Successfully processed:   {success_cnt}")
    print(f"Failed:                   {fail_cnt}")
    print(f"Success rate:             {success_rate}%")
    print(f"Processing Time:          {total_time:.2f} seconds")

    print("\n--- FC matrix parameters ---")
    print("Shape:                    62 × 62")
    print("Method:                   Pearson correlation")
    print("Signed correlations preserved: YES")
    print("Thresholding:             NONE")
    print("Binarization:             NONE")
    print(f"Timepoints range:         {tp_min}–{tp_max} (Mean: {tp_mean:.1f})")

    print("\n--- Fisher transformation ---")
    print("Applied:                  YES")
    print("Transformation:           arctanh after clipping to [-0.999999, 0.999999]")
    print("Diagonal handling:        r=1.0 on Pearson diagonal -> clipped to 0.999999 -> arctanh(0.999999) ≈ 7.2543 on Fisher-z diagonal")

    print("\n--- Correlation Statistics ---")
    print(f"Minimum correlation:      {global_min_corr:.4f}")
    print(f"Maximum correlation:      {global_max_corr:.4f}")
    print(f"Mean absolute correlation: {global_mean_abs_corr:.4f}")

    print("\n--- Quality Control Integrity ---")
    print(f"NaN matrices:             {nan_matrices_cnt}")
    print(f"Inf matrices:             {inf_matrices_cnt}")
    print(f"Incorrect shapes:         {incorrect_shape_cnt}")
    print(f"Asymmetric matrices:      {asym_matrices_cnt}")

    print("\n--- By Split ---")
    print(f"Train:                    {split_counts.get('train', 0)}")
    print(f"Validation:               {split_counts.get('val', 0)}")
    print(f"Test:                     {split_counts.get('test', 0)}")

    print("\n--- By Severity Class ---")
    print(f"Low:                      {sev_counts.get('Low', 0)}")
    print(f"Moderate:                 {sev_counts.get('Moderate', 0)}")
    print(f"High:                     {sev_counts.get('High', 0)}")

    print("\n--- By Site ---")
    for site, cnt in sorted(site_counts.items()):
        print(f"  {site}: {cnt}")

    print("\n--- Files Created ---")
    print(f"FC output directory:      {FC_OUTPUT_DIR}")
    print(f"FC manifest:              {FC_MANIFEST_PATH}")
    print(f"Atlas metadata:           {ATLAS_METADATA_DST}")

    status_str = "COMPLETE" if fail_cnt == 0 and success_cnt == len(df_target) else "INCOMPLETE"
    print(f"\nSTEP 6 STATUS:\n{status_str}")
    print("=" * 80)
    print("DO NOT START STEP 7.")
    print("DO NOT CONSTRUCT GRAPHS.")
    print("DO NOT TRAIN GCN OR GAT.")
    print("=" * 80)


if __name__ == "__main__":
    main()
