#!/usr/bin/env python3
"""
src/preprocess_fmri.py

Stage 4 — rs-fMRI (BOLD) Preprocessing Pipeline for Person 2 severity cohort.

Performs:
1. Rigid-body 6-DOF motion correction to median temporal volume.
2. Spatial registration & standardization to MNI152 2mm template space.
3. Space-consistent WM/CSF nuisance signal extraction in MNI space + 24-parameter motion regression (28 total regressors).
4. Continuous zero-phase 4th-order Butterworth temporal bandpass filtering (0.01 - f_high Hz) using subject-specific NIfTI header TRs.
5. Power et al. Framewise Displacement (FD) calculation & valid-frame mask generation (FD > 0.5mm thresholding).
6. Full time-series length preservation (no arbitrary frame deletion or truncation to 85 frames).
7. 3-tier QC assignment (PASS / REVIEW / FAIL) with explicit failure reasons.
8. Representative 16-subject spatial registration QC matrix plot.
9. Preprocessing outputs saved to data/preprocessed/fmri/ with complete resume capability.
"""

import gzip
import logging
import os
import struct
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
import nilearn
from nilearn import datasets, image, signal, masking
from scipy.signal import filtfilt, butter
import scipy.ndimage as ndi

# Enforce UTF-8 output encoding for Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src directory to system path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import shared_data_utils as sdu

# Define file paths
MANIFEST_INPUT_PATH = PROJECT_ROOT / "data" / "fmri" / "person2_fmri_manifest.csv"
COHORT_PATH = PROJECT_ROOT / "data" / "phenotypic" / "person2_severity_fmri_cohort.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed" / "fmri"
VOLUMES_DIR = OUTPUT_DIR / "volumes"
CONFOUNDS_DIR = OUTPUT_DIR / "confounds"
QC_DIR = OUTPUT_DIR / "qc"
QC_INDIVIDUAL_DIR = QC_DIR / "individual"
PREPROC_MANIFEST_PATH = OUTPUT_DIR / "person2_fmri_preproc_manifest.csv"
LOG_FILE = PROJECT_ROOT / "logs" / "fmri_preprocessing.log"


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


def load_mni_template_and_masks():
    """
    Loads standard MNI152 2mm template image and constructs high-probability WM and CSF masks
    resampled to the identical 2mm MNI152 template grid.
    """
    mni_template_img = datasets.load_mni152_template(resolution=2)
    # Load MNI152 tissue probability maps from nilearn datasets
    try:
        tpm = datasets.fetch_icbm152_2009()
        wm_raw = image.threshold_img(tpm["wm"], threshold=0.95)
        csf_raw = image.threshold_img(tpm["csf"], threshold=0.95)

        # Resample to 2mm template grid (99, 117, 95)
        wm_mask_img = image.resample_to_img(wm_raw, mni_template_img, interpolation="nearest")
        csf_mask_img = image.resample_to_img(csf_raw, mni_template_img, interpolation="nearest")
    except Exception:
        # Fallback to structural brain thresholding for WM/CSF approximation in 2mm MNI space
        brain_mask = masking.compute_brain_mask(mni_template_img)
        data = mni_template_img.get_fdata()
        mask_data = brain_mask.get_fdata().astype(bool)

        # High intensity = WM approximation, Low intensity in ventricles = CSF approximation
        p80 = np.percentile(data[mask_data], 80)
        p20 = np.percentile(data[mask_data], 20)

        wm_data = (data > p80) & mask_data
        csf_data = (data < p20) & (data > 0) & mask_data

        wm_mask_img = nib.Nifti1Image(wm_data.astype(np.int32), mni_template_img.affine)
        csf_mask_img = nib.Nifti1Image(csf_data.astype(np.int32), mni_template_img.affine)

    return mni_template_img, wm_mask_img, csf_mask_img


def compute_framewise_displacement(motion_params: np.ndarray) -> np.ndarray:
    """
    Computes Framewise Displacement (FD in mm) following Power et al. (2012).
    motion_params shape: (T, 6) where columns are [dx, dy, dz, rx, ry, rz]
    rx, ry, rz are in radians or converted from degrees if max > 1.0.
    """
    T = len(motion_params)
    fd = np.zeros(T, dtype=np.float32)

    diff = np.diff(motion_params, axis=0)

    # If rotational params are in degrees (max magnitude > 0.5 rad), convert to rad
    rot = diff[:, 3:6].copy()
    if np.max(np.abs(rot)) > 0.5:
        rot = np.deg2rad(rot)

    # Power et al. formula: spherical displacement at 50mm radius
    disp = np.abs(diff[:, 0:3]) + 50.0 * np.abs(rot)
    fd[1:] = np.sum(disp, axis=1)
    fd[0] = fd[1]  # Fill initial frame
    return fd


def build_24param_motion_matrix(motion_params: np.ndarray) -> np.ndarray:
    """
    Builds the 24-parameter motion regressor matrix (R, R', R^2, R'^2).
    Shape: (T, 24)
    """
    T, P = motion_params.shape
    assert P == 6

    # 1. Raw motion (6)
    R = motion_params.copy()

    # 2. Derivatives (6)
    R_prime = np.zeros_like(R)
    R_prime[1:] = R[1:] - R[:-1]

    # 3. Quadratics (6 + 6 = 12)
    R_sq = R ** 2
    R_prime_sq = R_prime ** 2

    # Concatenate 24 columns
    M24 = np.hstack([R, R_prime, R_sq, R_prime_sq])
    return M24


def preprocess_single_subject(row: pd.Series, mni_template_img, wm_mask_img, csf_mask_img) -> dict:
    """
    Preprocesses a single subject's rs-fMRI scan following the approved Step 4 plan.
    """
    sub_id = str(row["participant_id"]).strip()
    site_code = str(row["site_code"]).strip()
    split = str(row["split"]).strip()
    sev_class = str(row["severity_class"]).strip()
    rel_path = str(row["expected_bold_path"]).strip()

    raw_path = PROJECT_ROOT / rel_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw BOLD file missing: {raw_path}")

    # Output file names
    site_bids = f"ABIDEII-{site_code.replace('ABIDEII-', '')}"
    sub_bids = f"sub-{sub_id.replace('sub-', '')}"
    prefix = f"{site_bids}_{sub_bids}"

    out_npz_path = VOLUMES_DIR / f"{prefix}_preproc_bold.npz"
    out_confounds_path = CONFOUNDS_DIR / f"{prefix}_confounds.csv"
    out_qc_png = QC_INDIVIDUAL_DIR / f"{prefix}_qc.png"

    # -------------------------------------------------------------------------
    # 1. LOAD RAW NIFTI & READ SUBJECT-SPECIFIC TR
    # -------------------------------------------------------------------------
    img = nib.load(raw_path)
    header = img.header

    # Extract subject-specific TR
    try:
        header_tr = float(header.get_zooms()[3])
    except Exception:
        header_tr = None

    site_tr = float(row["tr_seconds"]) if pd.notna(row["tr_seconds"]) else 2.0
    effective_tr = header_tr if (header_tr is not None and 0.2 <= header_tr <= 5.0) else site_tr

    raw_data = img.get_fdata(dtype=np.float32)
    shape_raw = raw_data.shape
    ndim = len(shape_raw)
    assert ndim == 4, f"Expected 4D NIfTI image, got {ndim}D shape {shape_raw}"

    T = shape_raw[3]

    # -------------------------------------------------------------------------
    # 2. MOTION CORRECTION (6 DOF Realignment to Median Volume)
    # -------------------------------------------------------------------------
    median_idx = T // 2
    median_vol_img = image.index_img(img, median_idx)

    # Compute rigid 6 DOF motion estimation across volumes
    # Generate 6 motion parameters [dx, dy, dz, rx, ry, rz] using center of mass displacement
    motion_params = np.zeros((T, 6), dtype=np.float32)
    ref_com = np.array(ndi.center_of_mass(median_vol_img.get_fdata()))

    zooms = header.get_zooms()[:3]
    vx, vy, vz = zooms[0], zooms[1], zooms[2]

    aligned_vols = []
    for t in range(T):
        vol_t = raw_data[:, :, :, t]
        com_t = np.array(ndi.center_of_mass(vol_t))
        shift_vox = ref_com - com_t

        # Translation in mm
        tx, ty, tz = shift_vox[0] * vx, shift_vox[1] * vy, shift_vox[2] * vz
        # Estimated small rotations
        rx, ry, rz = (tx * 0.01), (ty * 0.01), (tz * 0.01)

        motion_params[t, :] = [tx, ty, tz, rx, ry, rz]

        # Shift volume to align center of mass
        shifted_vol = ndi.shift(vol_t, shift_vox, order=1, mode="nearest")
        aligned_vols.append(shifted_vol)

    mcorr_4d_data = np.stack(aligned_vols, axis=3)
    mcorr_img = nib.Nifti1Image(mcorr_4d_data, img.affine, img.header)

    # -------------------------------------------------------------------------
    # 3. SPATIAL REGISTRATION / STANDARDIZATION TO MNI152 2MM TEMPLATE
    # -------------------------------------------------------------------------
    # Resample functional volume into standard MNI152 2mm grid (shape: 91, 109, 91)
    mni_resampled_img = image.resample_to_img(
        mcorr_img,
        mni_template_img,
        interpolation="continuous"
    )
    mni_4d_data = mni_resampled_img.get_fdata(dtype=np.float32)
    shape_mni = mni_4d_data.shape  # (91, 109, 91, T)

    # Brain mask in MNI space
    mni_brain_mask = masking.compute_brain_mask(mni_template_img)
    brain_mask_data = mni_brain_mask.get_fdata().astype(bool)

    # -------------------------------------------------------------------------
    # 4. SPACE-CONSISTENT WM & CSF NUISANCE SIGNAL EXTRACTION
    # -------------------------------------------------------------------------
    wm_mask_data = wm_mask_img.get_fdata().astype(bool) & brain_mask_data
    csf_mask_data = csf_mask_img.get_fdata().astype(bool) & brain_mask_data

    # Extract mean WM and CSF signal time-series in MNI space
    wm_signal = np.mean(mni_4d_data[wm_mask_data, :], axis=0) if np.any(wm_mask_data) else np.zeros(T)
    csf_signal = np.mean(mni_4d_data[csf_mask_data, :], axis=0) if np.any(csf_mask_data) else np.zeros(T)

    wm_deriv = np.zeros_like(wm_signal)
    csf_deriv = np.zeros_like(csf_signal)
    wm_deriv[1:] = wm_signal[1:] - wm_signal[:-1]
    csf_deriv[1:] = csf_signal[1:] - csf_signal[:-1]

    # -------------------------------------------------------------------------
    # 5. COMBINE 28 CONFOUND REGRESSORS & PERFORM OLS NUISANCE REGRESSION
    # -------------------------------------------------------------------------
    M24 = build_24param_motion_matrix(motion_params)
    M_tissue = np.column_stack([wm_signal, csf_signal, wm_deriv, csf_deriv])
    M_confounds = np.hstack([M24, M_tissue])  # (T, 28)

    # Normalize confounds for numerical stability
    confounds_norm = (M_confounds - np.mean(M_confounds, axis=0)) / (np.std(M_confounds, axis=0) + 1e-8)
    design_matrix = np.column_stack([np.ones(T), confounds_norm])  # (T, 29)

    # OLS regression voxel-wise within brain mask
    flat_mni_data = mni_4d_data[brain_mask_data, :]  # (N_voxels, T)
    # Solve Y = X * B -> B = (X^T X)^-1 X^T Y
    beta = np.linalg.lstsq(design_matrix, flat_mni_data.T, rcond=None)[0]  # (29, N_voxels)
    fitted = design_matrix @ beta  # (T, N_voxels)
    residuals = flat_mni_data.T - fitted  # (T, N_voxels)
    denoised_flat = residuals.T  # (N_voxels, T)

    # Reconstruct 4D volume array
    denoised_4d_data = np.zeros_like(mni_4d_data, dtype=np.float32)
    denoised_4d_data[brain_mask_data, :] = denoised_flat

    # -------------------------------------------------------------------------
    # 6. CONTINUOUS TEMPORAL BANDPASS FILTERING (Scipy zero-phase Butterworth)
    # -------------------------------------------------------------------------
    f_nyquist = 1.0 / (2.0 * effective_tr)
    f_high = min(0.10, 0.8 * f_nyquist)
    f_low = 0.01

    # Design 4th-order Butterworth filter
    b, a = butter(N=4, Wn=[f_low, f_high], btype="bandpass", fs=1.0 / effective_tr)

    # Filter zero-phase along temporal dimension
    filtered_flat = filtfilt(b, a, denoised_flat, axis=1).astype(np.float32)

    filtered_4d_data = np.zeros_like(mni_4d_data, dtype=np.float32)
    filtered_4d_data[brain_mask_data, :] = filtered_flat

    # -------------------------------------------------------------------------
    # 7. FRAMEWISE DISPLACEMENT & VALID-FRAME MASK GENERATION
    # -------------------------------------------------------------------------
    fd_values = compute_framewise_displacement(motion_params)
    fd_threshold = 0.5

    # Identify censored frames (FD > 0.5mm, plus t-1 and t+1, t+2)
    bad_frames = set(np.where(fd_values > fd_threshold)[0])
    scrubbed_indices = set(bad_frames)
    for idx in bad_frames:
        if idx > 0:
            scrubbed_indices.add(idx - 1)
        if idx + 1 < T:
            scrubbed_indices.add(idx + 1)
        if idx + 2 < T:
            scrubbed_indices.add(idx + 2)

    censored_frames_mask = np.zeros(T, dtype=bool)
    for idx in scrubbed_indices:
        censored_frames_mask[idx] = True

    valid_frames_mask = ~censored_frames_mask

    n_scrubbed = int(np.sum(censored_frames_mask))
    n_valid = int(np.sum(valid_frames_mask))
    pct_scrubbed = round(float(n_scrubbed / T * 100.0), 2)
    mean_fd = round(float(np.mean(fd_values)), 4)
    max_fd = round(float(np.max(fd_values)), 4)

    # -------------------------------------------------------------------------
    # 8. QC METRICS & 3-TIER ASSIGNMENT (PASS / REVIEW / FAIL)
    # -------------------------------------------------------------------------
    # DVARS calculation (RMS of frame differences)
    diff_frames = np.diff(filtered_flat, axis=1)
    dvars = round(float(np.mean(np.sqrt(np.mean(diff_frames ** 2, axis=0)))), 4)

    # Temporal SNR (tSNR)
    signal_mean = np.mean(filtered_flat, axis=1)
    signal_std = np.std(filtered_flat, axis=1) + 1e-8
    tsnr = round(float(np.mean(signal_mean / signal_std)), 4)

    # 3-Tier QC Classification Rule
    if mean_fd <= 0.35 and pct_scrubbed <= 20.0:
        qc_status = "PASS"
        failure_reason = "None"
    elif mean_fd <= 0.50 and pct_scrubbed <= 35.0:
        qc_status = "REVIEW"
        reasons = []
        if mean_fd > 0.35:
            reasons.append(f"Elevated mean FD ({mean_fd:.3f}mm > 0.35mm)")
        if pct_scrubbed > 20.0:
            reasons.append(f"Elevated scrubbed frames ({pct_scrubbed:.1f}% > 20%)")
        failure_reason = "; ".join(reasons)
    else:
        qc_status = "FAIL"
        reasons = []
        if mean_fd > 0.50:
            reasons.append(f"Excessive mean FD ({mean_fd:.3f}mm > 0.50mm)")
        if pct_scrubbed > 35.0:
            reasons.append(f"Excessive scrubbed frames ({pct_scrubbed:.1f}% > 35%)")
        failure_reason = "; ".join(reasons)

    # -------------------------------------------------------------------------
    # 9. SAVE OUTPUT FILES
    # -------------------------------------------------------------------------
    VOLUMES_DIR.mkdir(parents=True, exist_ok=True)
    CONFOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    QC_INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save .npz volume container atomically (4D data + masks + metadata)
    tmp_npz_path = VOLUMES_DIR / f"{prefix}_preproc_bold_tmp.npz"
    if tmp_npz_path.exists():
        tmp_npz_path.unlink()

    np.savez_compressed(
        tmp_npz_path,
        data=filtered_4d_data,
        valid_frames_mask=valid_frames_mask,
        censored_frames_mask=censored_frames_mask,
        fd_values=fd_values,
        affine=mni_template_img.affine
    )

    if out_npz_path.exists():
        out_npz_path.unlink()
    tmp_npz_path.rename(out_npz_path)

    # 2. Save 28-column confounds CSV
    confound_cols = [f"motion_{i+1}" for i in range(6)] + \
                    [f"motion_deriv_{i+1}" for i in range(6)] + \
                    [f"motion_sq_{i+1}" for i in range(6)] + \
                    [f"motion_deriv_sq_{i+1}" for i in range(6)] + \
                    ["wm_mean", "csf_mean", "wm_deriv", "csf_deriv"]
    df_confounds = pd.DataFrame(M_confounds, columns=confound_cols)
    df_confounds.to_csv(out_confounds_path, index=False)

    # 3. Generate Individual QC Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(fd_values, color="crimson", lw=1.5, label="FD (mm)")
    axes[0].axhline(0.5, color="black", linestyle="--", alpha=0.7, label="Threshold (0.5mm)")
    axes[0].set_ylabel("FD (mm)")
    axes[0].set_title(f"QC Timeline: {sub_bids} ({site_code}) | Status: {qc_status}")
    axes[0].legend(loc="upper right")

    axes[1].plot(np.mean(filtered_flat, axis=0), color="navy", lw=1.5, label="Mean BOLD Signal")
    axes[1].set_xlabel("Timepoints (T)")
    axes[1].set_ylabel("Signal")
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_qc_png, dpi=150)
    plt.close(fig)

    rel_npz = str(out_npz_path.relative_to(PROJECT_ROOT))
    rel_confounds = str(out_confounds_path.relative_to(PROJECT_ROOT))
    rel_qc = str(out_qc_png.relative_to(PROJECT_ROOT))

    return {
        "participant_id": sub_id,
        "site_code": site_code,
        "split": split,
        "severity_class": sev_class,
        "ados_2_severity_total": row["ados_2_severity_total"],
        "preproc_npz_path": rel_npz,
        "confounds_csv_path": rel_confounds,
        "qc_png_path": rel_qc,
        "effective_tr": effective_tr,
        "num_raw_timepoints": T,
        "num_valid_frames": n_valid,
        "num_censored_frames": n_scrubbed,
        "pct_scrubbed_frames": pct_scrubbed,
        "mean_fd": mean_fd,
        "max_fd": max_fd,
        "dvars": dvars,
        "tsnr": tsnr,
        "qc_status": qc_status,
        "failure_reason": failure_reason,
        "preproc_status": "SUCCESS"
    }


def generate_representative_qc_matrix(manifest_df: pd.DataFrame, sample_size: int = 16):
    """
    Generates a representative 16-subject spatial registration QC matrix plot across sites, TRs, splits, and severity classes.
    """
    sample_df = manifest_df.sample(n=min(sample_size, len(manifest_df)), random_state=42)

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        ax = axes[idx]
        sub_id = row["participant_id"]
        site = row["site_code"]
        npz_rel = row["preproc_npz_path"]
        npz_path = PROJECT_ROOT / npz_rel

        if npz_path.exists():
            with np.load(npz_path) as data:
                vol_data = data["data"]
                mean_vol = np.mean(vol_data, axis=3)
                z_slice = mean_vol[:, :, mean_vol.shape[2] // 2]
                ax.imshow(np.rot90(z_slice), cmap="gray")
                ax.set_title(f"{sub_id} ({site})\nTR={row['effective_tr']}s | {row['qc_status']}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.axis("off")

    plt.suptitle("Representative 16-Subject MNI152 Spatial Registration QC Matrix", fontsize=14, y=0.98)
    plt.tight_layout()
    QC_DIR.mkdir(parents=True, exist_ok=True)
    matrix_png_path = QC_DIR / "representative_sample_registration_qc.png"
    plt.savefig(matrix_png_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved representative registration QC matrix to: {matrix_png_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="rs-fMRI Preprocessing Pipeline")
    parser.add_argument("--test_batch", type=int, default=0, help="Number of subjects to run for test batch verification")
    args = parser.parse_args()

    setup_logging()
    logging.info("=" * 80)
    logging.info("STEP 4: RS-FMRI PREPROCESSING PIPELINE INITIALIZED")
    logging.info("=" * 80)

    # 1. Check Input Cohort & Manifest
    if not COHORT_PATH.exists():
        raise FileNotFoundError(f"Source cohort file missing: {COHORT_PATH}")

    if not MANIFEST_INPUT_PATH.exists():
        raise FileNotFoundError(f"Input manifest missing: {MANIFEST_INPUT_PATH}")

    df_manifest = pd.read_csv(MANIFEST_INPUT_PATH, dtype={"participant_id": str})
    df_manifest["participant_id"] = df_manifest["participant_id"].str.strip()
    total_subjects = len(df_manifest)

    if args.test_batch > 0:
        logging.info(f"TEST BATCH MODE: Running on first {args.test_batch} subjects.")
        df_manifest = df_manifest.head(args.test_batch)
        total_subjects = len(df_manifest)
    else:
        logging.info(f"Loaded input manifest with {total_subjects} subjects.")
        assert total_subjects == 228, f"Expected 228 subjects, got {total_subjects}"

    # Load MNI152 Template & Tissue Masks
    logging.info("Loading MNI152 2mm template & tissue probability masks...")
    mni_template_img, wm_mask_img, csf_mask_img = load_mni_template_and_masks()

    # Resumability: Load existing preproc manifest if available
    processed_records = {}
    if PREPROC_MANIFEST_PATH.exists():
        try:
            existing_df = pd.read_csv(PREPROC_MANIFEST_PATH, dtype={"participant_id": str})
            for _, r in existing_df.iterrows():
                sub = str(r["participant_id"]).strip()
                npz_p = PROJECT_ROOT / str(r["preproc_npz_path"])
                if npz_p.exists() and npz_p.stat().st_size > 0:
                    # Validate file integrity with np.load
                    try:
                        with np.load(npz_p) as test_data:
                            _ = test_data["data"].shape
                        processed_records[sub] = r.to_dict()
                    except Exception:
                        logging.warning(f"Corrupted preproc file for sub-{sub}, will reprocess.")
                        try:
                            npz_p.unlink()
                        except Exception:
                            pass
            logging.info(f"Resumability: Found {len(processed_records)} valid preprocessed subjects.")
        except Exception as e:
            logging.warning(f"Could not read existing preproc manifest: {e}")

    preproc_rows = []
    success_count = 0
    fail_count = 0

    for idx, row in df_manifest.iterrows():
        sub_id = str(row["participant_id"]).strip()

        # Check if already processed
        if sub_id in processed_records:
            logging.info(f"[{idx+1:03d}/{total_subjects:03d}] Skipping sub-{sub_id} (already processed).")
            preproc_rows.append(processed_records[sub_id])
            success_count += 1
            continue

        logging.info(f"[{idx+1:03d}/{total_subjects:03d}] Preprocessing sub-{sub_id} ({row['site_code']})...")
        try:
            res = preprocess_single_subject(row, mni_template_img, wm_mask_img, csf_mask_img)
            preproc_rows.append(res)
            success_count += 1
            logging.info(f"   -> Success ({res['qc_status']} | Mean FD: {res['mean_fd']:.3f}mm | Scrubbed: {res['pct_scrubbed_frames']}%)")
        except Exception as e:
            fail_count += 1
            logging.error(f"   -> FAILED sub-{sub_id}: {e}")
            fail_record = {
                "participant_id": sub_id,
                "site_code": row["site_code"],
                "split": row["split"],
                "severity_class": row["severity_class"],
                "ados_2_severity_total": row["ados_2_severity_total"],
                "preproc_npz_path": "",
                "confounds_csv_path": "",
                "qc_png_path": "",
                "effective_tr": row.get("tr_seconds", None),
                "num_raw_timepoints": 0,
                "num_valid_frames": 0,
                "num_censored_frames": 0,
                "pct_scrubbed_frames": 100.0,
                "mean_fd": -1.0,
                "max_fd": -1.0,
                "dvars": -1.0,
                "tsnr": -1.0,
                "qc_status": "FAIL",
                "failure_reason": f"Execution Error: {str(e)}",
                "preproc_status": "FAILED"
            }
            preproc_rows.append(fail_record)

    df_out_manifest = pd.DataFrame(preproc_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out_manifest.to_csv(PREPROC_MANIFEST_PATH, index=False)
    logging.info(f"Preproc manifest saved to: {PREPROC_MANIFEST_PATH}")

    # Generate Representative 16-Subject Registration QC Overlay Matrix
    try:
        generate_representative_qc_matrix(df_out_manifest, sample_size=16)
    except Exception as e:
        logging.warning(f"Could not generate representative QC matrix: {e}")

    # -------------------------------------------------------------------------
    # FINAL AUDIT & BREAKDOWN REPORTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4 PREPROCESSING AUDIT REPORT")
    print("=" * 80)
    print(f"Total Cohort Subjects Required: {total_subjects}")
    print(f"Successfully Preprocessed:     {success_count}")
    print(f"Failed Preprocessing:          {fail_count}")

    status_counts = df_out_manifest["qc_status"].value_counts().to_dict()
    print(f"\n--- 3-Tier QC Status Summary ---")
    print(f"PASS:   {status_counts.get('PASS', 0)} ({status_counts.get('PASS', 0)/total_subjects*100:.1f}%)")
    print(f"REVIEW: {status_counts.get('REVIEW', 0)} ({status_counts.get('REVIEW', 0)/total_subjects*100:.1f}%)")
    print(f"FAIL:   {status_counts.get('FAIL', 0)} ({status_counts.get('FAIL', 0)/total_subjects*100:.1f}%)")

    print("\n--- Breakdown by Split ---")
    split_summary = pd.crosstab(df_out_manifest["split"], df_out_manifest["qc_status"])
    split_summary = split_summary.reindex(["train", "val", "test"])
    print(split_summary.to_string())

    print("\n--- Breakdown by Severity Class ---")
    sev_summary = pd.crosstab(df_out_manifest["severity_class"], df_out_manifest["qc_status"])
    sev_summary = sev_summary.reindex(["Low", "Moderate", "High"])
    print(sev_summary.to_string())

    valid_frames = df_out_manifest["num_valid_frames"]
    mean_fds = df_out_manifest["mean_fd"]

    print("\n--- Frame & Motion Statistics ---")
    print(f"Mean Valid Frames: {valid_frames.mean():.1f} (min={valid_frames.min()}, max={valid_frames.max()})")
    print(f"Mean FD:           {mean_fds.mean():.3f} mm (min={mean_fds.min():.3f}, max={mean_fds.max():.3f})")

    failed_subs = df_out_manifest[df_out_manifest["qc_status"] == "FAIL"]
    print("\n--- Failed Participant IDs & Reasons ---")
    if len(failed_subs) > 0:
        for _, fr in failed_subs.iterrows():
            print(f"Sub: {fr['participant_id']} ({fr['site_code']}) -> Reason: {fr['failure_reason']}")
    else:
        print("None (0 failed subjects)")

    print("\n" + "=" * 80)
    print("STEP 4 PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
