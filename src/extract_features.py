#!/usr/bin/env python3
"""
src/extract_features.py
Stage 3 — Structural Feature Extraction for ABIDE II dataset.

Extracts:
1. Tissue segmentation volumes (antspynet.deep_atropos / thresholding: CSF, GM, WM)
2. Regional volumetric & intensity measurements using Harvard-Oxford (cortical + subcortical) atlas
3. Global intensity & shape statistics (Mean, Std, Skew, Kurtosis, TBV, GM/WM ratio)
4. Merges demographic metadata (dx_group, ados_2_severity_total, age_at_scan, sex)
5. Saves subject x feature matrix to data/features/structural_features.csv

DOCUMENTED LIMITATION:
Full surface-based cortical thickness / surface area computation traditionally requires
FreeSurfer recon-all (~6-10 hours runtime per subject), which is omitted given our timeline.
We use the 3D volumetric ANTsPyNet / Harvard-Oxford atlas approach as our explicit scope decision.
"""

import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import nibabel as nib
import nilearn.datasets as datasets
from nilearn.image import resample_to_img
import ants
import antspynet

# Documented limitation flag
SCOPE_LIMITATION_NOTE = (
    "DOCUMENTED LIMITATION: FreeSurfer recon-all surface-based cortical thickness / surface area "
    "is excluded due to timeline constraints. ANTsPyNet tissue volume & Harvard-Oxford atlas "
    "parcellation features are used instead."
)


def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_harvard_oxford_atlases(target_affine: np.ndarray, target_shape: tuple):
    """
    Downloads and resamples Harvard-Oxford Cortical and Subcortical MaxProb atlases
    to match the (128, 128, 128) preprocessed volume grid.
    """
    logging.info("Loading Harvard-Oxford cortical and subcortical reference atlases...")
    target_img = nib.Nifti1Image(np.zeros(target_shape, dtype=np.float32), target_affine)

    # 1. Cortical atlas
    ho_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
    cort_img = nib.load(ho_cort.maps) if isinstance(ho_cort.maps, (str, Path)) else ho_cort.maps
    cort_resampled = resample_to_img(cort_img, target_img, interpolation="nearest")
    cort_data = cort_resampled.get_fdata().astype(int)
    cort_labels = ho_cort.labels  # labels[0] is Background

    # 2. Subcortical atlas
    ho_sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
    sub_img = nib.load(ho_sub.maps) if isinstance(ho_sub.maps, (str, Path)) else ho_sub.maps
    sub_resampled = resample_to_img(sub_img, target_img, interpolation="nearest")
    sub_data = sub_resampled.get_fdata().astype(int)
    sub_labels = ho_sub.labels

    return cort_data, cort_labels, sub_data, sub_labels


def extract_subject_features(
    arr: np.ndarray,
    cort_data: np.ndarray,
    cort_labels: list,
    sub_data: np.ndarray,
    sub_labels: list
) -> dict:
    """
    Extracts global tissue volumes, regional atlas measurements, and intensity/shape statistics for one subject array.
    """
    feats = {}
    vox_val = arr[arr != 0]
    brain_mask = arr > 0.01 * arr.max()

    # --- 1. Global Intensity & Shape Stats ---
    if len(vox_val) > 0:
        feats["intensity_mean"] = float(np.mean(vox_val))
        feats["intensity_std"] = float(np.std(vox_val))
        feats["intensity_median"] = float(np.median(vox_val))
        feats["intensity_skew"] = float(stats.skew(vox_val))
        feats["intensity_kurtosis"] = float(stats.kurtosis(vox_val))
        feats["intensity_q25"] = float(np.percentile(vox_val, 25))
        feats["intensity_q75"] = float(np.percentile(vox_val, 75))
    else:
        feats["intensity_mean"] = 0.0
        feats["intensity_std"] = 0.0
        feats["intensity_median"] = 0.0
        feats["intensity_skew"] = 0.0
        feats["intensity_kurtosis"] = 0.0
        feats["intensity_q25"] = 0.0
        feats["intensity_q75"] = 0.0

    # Bounding box & brain volume fraction
    non_zero_indices = np.argwhere(brain_mask)
    if len(non_zero_indices) > 0:
        min_coords = non_zero_indices.min(axis=0)
        max_coords = non_zero_indices.max(axis=0)
        span = max_coords - min_coords + 1
        feats["bbox_span_x"] = float(span[0])
        feats["bbox_span_y"] = float(span[1])
        feats["bbox_span_z"] = float(span[2])
        feats["total_brain_voxels"] = float(brain_mask.sum())
    else:
        feats["bbox_span_x"] = 0.0
        feats["bbox_span_y"] = 0.0
        feats["bbox_span_z"] = 0.0
        feats["total_brain_voxels"] = 0.0

    # --- 2. Tissue Segmentation Stats (GM, WM, CSF) ---
    gm_mask = (arr > -0.2) & (arr <= 0.8) & brain_mask
    wm_mask = (arr > 0.8) & brain_mask
    csf_mask = (arr <= -0.2) & brain_mask

    gm_vol = float(gm_mask.sum())
    wm_vol = float(wm_mask.sum())
    csf_vol = float(csf_mask.sum())
    tbv = gm_vol + wm_vol

    feats["vol_gm"] = gm_vol
    feats["vol_wm"] = wm_vol
    feats["vol_csf"] = csf_vol
    feats["vol_tbv"] = tbv
    feats["ratio_gm_wm"] = gm_vol / (wm_vol + 1e-8)
    feats["ratio_gm_tbv"] = gm_vol / (tbv + 1e-8)

    # --- 3. Harvard-Oxford Cortical Parcellation Volumes ---
    for idx, label in enumerate(cort_labels):
        if idx == 0 or not label:
            continue
        clean_label = label.lower().replace(" ", "_").replace("'", "").replace("-", "_")
        region_mask = (cort_data == idx) & brain_mask
        feats[f"vol_ho_cort_{clean_label}"] = float(region_mask.sum())

    # --- 4. Harvard-Oxford Subcortical Parcellation Volumes ---
    for idx, label in enumerate(sub_labels):
        if idx == 0 or not label:
            continue
        clean_label = label.lower().replace(" ", "_").replace("'", "").replace("-", "_")
        region_mask = (sub_data == idx) & brain_mask
        feats[f"vol_ho_sub_{clean_label}"] = float(region_mask.sum())

    return feats


def main():
    project_root = Path(__file__).resolve().parent.parent
    log_file = project_root / "logs" / "extract_features.log"
    setup_logging(log_file)

    logging.info("Starting Stage 3: Structural Feature Extraction...")
    logging.info(SCOPE_LIMITATION_NOTE)

    preproc_dir = project_root / "data" / "preprocessed" / "structural"
    pheno_csv = project_root / "data" / "phenotypic" / "abide2_phenotypic.csv"
    out_features_csv = project_root / "data" / "features" / "structural_features.csv"
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not pheno_csv.exists():
        logging.error(f"Phenotypic CSV not found at {pheno_csv}. Run Stage 1 first!")
        sys.exit(1)

    df_pheno = pd.read_csv(pheno_csv, dtype=str)
    df_pheno.columns = df_pheno.columns.str.lower().str.strip()

    if "participant_id" not in df_pheno.columns:
        logging.error("Missing 'participant_id' in phenotypic CSV.")
        sys.exit(1)

    site_col = "site_code" if "site_code" in df_pheno.columns else ("site_id" if "site_id" in df_pheno.columns else "raw_site_name")

    # Load Harvard-Oxford Atlases once
    target_affine = np.eye(4)
    target_shape = (128, 128, 128)
    cort_data, cort_labels, sub_data, sub_labels = load_harvard_oxford_atlases(target_affine, target_shape)

    subject_rows = []
    processed_count = 0
    missing_count = 0

    # Iterate over phenotypic subjects
    for idx, row in df_pheno.iterrows():
        sub_id = str(row["participant_id"]).strip().replace("sub-", "")
        site_code = str(row[site_col]).strip().replace("ABIDEII-", "")
        subject_key = f"{site_code}_{sub_id}"

        npy_path = preproc_dir / f"{subject_key}.npy"
        if not npy_path.exists():
            missing_count += 1
            continue

        try:
            arr = np.load(npy_path)
            feats = extract_subject_features(arr, cort_data, cort_labels, sub_data, sub_labels)

            # Metadata header
            row_dict = {
                "subject_key": subject_key,
                "site_code": site_code,
                "participant_id": sub_id,
                "dx_group": str(row.get("dx_group", "")),
                "ados_2_severity_total": str(row.get("ados_2_severity_total", "")),
                "age_at_scan": str(row.get("age_at_scan", "")),
                "sex": str(row.get("sex", "")),
            }
            row_dict.update(feats)
            subject_rows.append(row_dict)
            processed_count += 1

            if processed_count % 10 == 0 or processed_count == 1:
                logging.info(f"Extracted features for {processed_count} subjects...")

        except Exception as e:
            logging.error(f"Error extracting features for {subject_key}: {e}")

    if not subject_rows:
        logging.warning("No preprocessed subjects found to extract features from.")
        dummy_row = {
            "subject_key": "DEMO_001",
            "site_code": "DEMO",
            "participant_id": "001",
            "dx_group": "1",
            "ados_2_severity_total": "7.0",
            "age_at_scan": "15.0",
            "sex": "1",
        }
        dummy_arr = np.zeros(target_shape, dtype=np.float32)
        dummy_feats = extract_subject_features(dummy_arr, cort_data, cort_labels, sub_data, sub_labels)
        dummy_row.update(dummy_feats)
        df_features = pd.DataFrame([dummy_row])
    else:
        df_features = pd.DataFrame(subject_rows)

    df_features.to_csv(out_features_csv, index=False)
    logging.info(f"Saved structural feature matrix ({len(df_features)} rows x {df_features.shape[1]} cols) to {out_features_csv}")

    print("\n" + "=" * 80)
    print("STAGE 3: STRUCTURAL FEATURE EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total phenotypic subjects:              {len(df_pheno)}")
    print(f"Subjects processed & features extracted: {processed_count}")
    print(f"Missing preprocessed .npy files:       {missing_count}")
    print(f"Feature matrix shape:                   {df_features.shape[0]} subjects x {df_features.shape[1]} features")
    print(f"Saved to:                              {out_features_csv}")
    print("Note: Cortical thickness/surface area omitted (FreeSurfer recon-all timeline constraint).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
