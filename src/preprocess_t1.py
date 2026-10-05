#!/usr/bin/env python3
"""
src/preprocess_t1.py
Stage 2 — T1 Preprocessing Pipeline for ABIDE II dataset.

Per subject:
- Downloads T1 scan from S3 (anat/ folder)
- Skips if already processed (maintains logs/t1_preprocessing_processed.log)
- N4 bias field correction (ANTsPy)
- Deep-learning skull-stripping (antspynet.brain_extraction)
- Registers to MNI152 template (affine registration)
- Resamples to fixed shape (128, 128, 128) & z-score normalizes brain voxels
- Saves preprocessed volume to data/preprocessed/structural/<subject_key>.npy
- Generates 3-slice (Axial, Sagittal, Coronal) QC PNG to results/qc_slices/
- Deletes raw downloaded NIfTI file after successful processing
"""

import argparse
import logging
import os
import sys
import shutil
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import ants
import antspynet

S3_BUCKET = "fcp-indi"
S3_BASE_PREFIX = "data/Projects/ABIDE2/RawData"


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


def load_processed_subjects(resume_log: Path) -> set:
    if not resume_log.exists():
        return set()
    with open(resume_log, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def mark_subject_processed(resume_log: Path, subject_key: str):
    resume_log.parent.mkdir(parents=True, exist_ok=True)
    with open(resume_log, "a", encoding="utf-8") as f:
        f.write(f"{subject_key}\n")


def download_raw_t1_s3(s3_client, site_code: str, participant_id: str, dest_file: Path) -> bool:
    """
    Downloads the raw anatomical T1 scan for a given site and participant ID from S3.
    """
    site_raw = site_code if site_code.startswith("ABIDEII-") else f"ABIDEII-{site_code}"
    sub_id_clean = str(participant_id).strip().replace("sub-", "")

    possible_prefixes = [
        f"{S3_BASE_PREFIX}/{site_raw}/sub-{sub_id_clean}/ses-1/anat/",
        f"{S3_BASE_PREFIX}/{site_raw}/sub-{sub_id_clean}/anat/",
        f"{S3_BASE_PREFIX}/{site_raw}/{sub_id_clean}/ses-1/anat/",
        f"{S3_BASE_PREFIX}/{site_raw}/{sub_id_clean}/anat/",
    ]

    paginator = s3_client.get_paginator("list_objects_v2")
    for prefix in possible_prefixes:
        try:
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                contents = page.get("Contents", [])
                for obj in contents:
                    key = obj["Key"]
                    if key.endswith(".nii.gz") or key.endswith(".nii"):
                        logging.info(f"Downloading s3://{S3_BUCKET}/{key} -> {dest_file}")
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        s3_client.download_file(S3_BUCKET, key, str(dest_file))
                        return True
        except Exception as e:
            logging.debug(f"Prefix {prefix} query issue: {e}")

    logging.warning(f"No anatomical T1 scan found on S3 for subject '{participant_id}' in site '{site_code}'.")
    return False


def generate_qc_slice_png(arr: np.ndarray, subject_key: str, out_png: Path):
    """
    Generates a 3-slice (Axial, Sagittal, Coronal) sanity check plot.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = arr.shape

    sagittal = arr[sx // 2, :, :]
    coronal = arr[:, sy // 2, :]
    axial = arr[:, :, sz // 2]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(np.rot90(sagittal), cmap="gray")
    axes[0].set_title(f"{subject_key} - Sagittal")
    axes[0].axis("off")

    axes[1].imshow(np.rot90(coronal), cmap="gray")
    axes[1].set_title(f"{subject_key} - Coronal")
    axes[1].axis("off")

    axes[2].imshow(np.rot90(axial), cmap="gray")
    axes[2].set_title(f"{subject_key} - Axial")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved QC 3-slice preview to {out_png}")


def preprocess_subject(
    raw_nii_path: Path,
    out_npy_path: Path,
    qc_png_path: Path,
    subject_key: str,
    mni_template: ants.ANTsImage
) -> bool:
    """
    Full ANTsPy T1 preprocessing pipeline per subject:
    1. Read raw NIfTI
    2. N4 Bias Field Correction
    3. Skull-stripping via antspynet.brain_extraction
    4. Affine Registration to MNI152
    5. Resample to (128, 128, 128) & z-score normalize
    """
    try:
        logging.info(f"[{subject_key}] Step 1/5: Loading raw image...")
        raw_img = ants.image_read(str(raw_nii_path))

        logging.info(f"[{subject_key}] Step 2/5: N4 Bias Field Correction...")
        n4_img = ants.n4_bias_field_correction(raw_img)

        logging.info(f"[{subject_key}] Step 3/5: Deep learning skull-stripping (antspynet)...")
        prob_mask = antspynet.brain_extraction(n4_img, modality="t1", verbose=False)
        binary_mask = ants.threshold_image(prob_mask, 0.5, 1.0, 1, 0)
        brain_img = n4_img * binary_mask

        logging.info(f"[{subject_key}] Step 4/5: Affine registration to MNI152 template...")
        reg = ants.registration(fixed=mni_template, moving=brain_img, type_of_transform="Affine")
        warped_brain = reg["warpedmovout"]

        logging.info(f"[{subject_key}] Step 5/5: Resampling to (128, 128, 128) & Z-score normalizing...")
        resampled_img = ants.resample_image(warped_brain, (128, 128, 128), use_voxels=True, interp_type=1)
        
        arr = resampled_img.numpy().astype(np.float32)
        brain_voxels = arr > 0.01 * arr.max()
        if brain_voxels.sum() > 0:
            mean_val = arr[brain_voxels].mean()
            std_val = arr[brain_voxels].std() + 1e-8
            arr_norm = np.zeros_like(arr, dtype=np.float32)
            arr_norm[brain_voxels] = (arr[brain_voxels] - mean_val) / std_val
        else:
            arr_norm = arr

        # Save preprocessed .npy file
        out_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy_path, arr_norm)
        logging.info(f"[{subject_key}] Saved preprocessed volume array to {out_npy_path} (shape: {arr_norm.shape})")

        # Generate QC slice visualization
        generate_qc_slice_png(arr_norm, subject_key, qc_png_path)
        return True

    except Exception as e:
        logging.error(f"[{subject_key}] Preprocessing error: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Stage 2 — T1 MRI Preprocessing Pipeline")
    parser.add_argument("--test_n", type=int, default=None, help="Number of test subjects to process for visual QC sanity check")
    parser.add_argument("--site", type=str, default=None, help="Filter processing to a specific site_code (e.g. NYU_1)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    log_file = project_root / "logs" / "t1_preprocessing.log"
    resume_log = project_root / "logs" / "t1_preprocessing_processed.log"
    setup_logging(log_file)

    pheno_csv = project_root / "data" / "phenotypic" / "abide2_phenotypic.csv"
    raw_dir = project_root / "data" / "raw" / "abide2_t1"
    preproc_dir = project_root / "data" / "preprocessed" / "structural"
    qc_dir = project_root / "results" / "qc_slices"

    if not pheno_csv.exists():
        logging.error(f"Phenotypic file missing at {pheno_csv}. Run Stage 1 (dataset_prep.py) first!")
        sys.exit(1)

    df_pheno = pd.read_csv(pheno_csv, dtype=str)
    df_pheno.columns = df_pheno.columns.str.lower().str.strip()

    # Determine site column robustly
    if "site_code" in df_pheno.columns:
        site_col = "site_code"
    elif "site_id" in df_pheno.columns:
        site_col = "site_id"
    elif "raw_site_name" in df_pheno.columns:
        site_col = "raw_site_name"
    else:
        site_col = df_pheno.columns[0]

    if args.site:
        clean_site = args.site.replace("ABIDEII-", "")
        df_pheno = df_pheno[df_pheno[site_col].astype(str).str.upper().str.contains(clean_site.upper())]
        logging.info(f"Filtered to site: {args.site} ({len(df_pheno)} subjects)")

    if args.test_n is not None:
        df_pheno = df_pheno.head(args.test_n)
        logging.info(f"Running in TEST mode on first {args.test_n} subjects.")

    processed_set = load_processed_subjects(resume_log)
    logging.info(f"Loaded {len(processed_set)} already processed subjects from resume log.")

    # Load MNI152 standard template once
    logging.info("Loading MNI152 template standard...")
    mni_template_path = ants.get_ants_data("mni")
    mni_template = ants.image_read(mni_template_path)

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    success_count = 0
    fail_count = 0

    for idx, row in df_pheno.iterrows():
        sub_id = str(row["participant_id"]).strip().replace("sub-", "")
        site_code = str(row[site_col]).strip().replace("ABIDEII-", "")
        subject_key = f"{site_code}_{sub_id}"

        out_npy_path = preproc_dir / f"{subject_key}.npy"
        qc_png_path = qc_dir / f"{subject_key}_qc.png"

        # Check resume state
        if subject_key in processed_set and out_npy_path.exists():
            logging.info(f"Skipping already processed subject '{subject_key}'")
            continue

        raw_nii_path = raw_dir / f"{subject_key}_raw.nii.gz"
        
        # Step 1: Download raw file if not present
        if not raw_nii_path.exists():
            download_ok = download_raw_t1_s3(s3_client, site_code, sub_id, raw_nii_path)
            if not download_ok:
                fail_count += 1
                continue

        # Step 2: Run preprocessing pipeline
        ok = preprocess_subject(raw_nii_path, out_npy_path, qc_png_path, subject_key, mni_template)

        # Step 3: Cleanup raw NIfTI file to save disk space
        if raw_nii_path.exists():
            try:
                raw_nii_path.unlink()
                logging.info(f"[{subject_key}] Removed raw file {raw_nii_path} to conserve disk space.")
            except Exception as e:
                logging.warning(f"Failed to delete raw file {raw_nii_path}: {e}")

        if ok:
            mark_subject_processed(resume_log, subject_key)
            processed_set.add(subject_key)
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 80)
    print("STAGE 2: PREPROCESSING RUN SUMMARY")
    print("=" * 80)
    print(f"Total target subjects processed in this run: {len(df_pheno)}")
    print(f"Successfully preprocessed: {success_count}")
    print(f"Failed / missing scans:    {fail_count}")
    if args.test_n:
        print(f"\nTest run complete! Please inspect QC images in: {qc_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
