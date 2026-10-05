#!/usr/bin/env python3
"""
src/person2_fmri_manifest.py

Stage 3 — rs-fMRI Data Manifest & Verification for Person 2 severity cohort.

This script:
1. Loads data/phenotypic/person2_severity_fmri_cohort.csv (228 ASD subjects).
2. Performs strict verification on cohort size, severity counts, and split counts.
3. Checks for duplicate participant IDs.
4. Generates the rs-fMRI BOLD manifest containing local expected paths, S3 URIs,
   file availability flags, site TRs, and NIfTI metadata (if files are present).
5. Does NOT download missing raw files or modify the input cohort CSV.
6. Saves output manifest to data/fmri/person2_fmri_manifest.csv.
"""

import gzip
import os
import struct
import sys
from pathlib import Path
import pandas as pd

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
COHORT_PATH = PROJECT_ROOT / "data" / "phenotypic" / "person2_severity_fmri_cohort.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "fmri"
MANIFEST_OUTPUT_PATH = OUTPUT_DIR / "person2_fmri_manifest.csv"
RAW_FMRI_DIR = PROJECT_ROOT / "data" / "raw" / "abide2_fmri"


def parse_nifti_header(file_path: Path) -> dict:
    """
    Parses a NIfTI-1 file (.nii or .nii.gz) binary header without external dependencies.
    Returns a dictionary with nifti_shape, num_timepoints, voxel_dimensions, and header TR.
    """
    try:
        if file_path.suffix == ".gz" or str(file_path).endswith(".nii.gz"):
            with gzip.open(file_path, "rb") as f:
                header_bytes = f.read(348)
        else:
            with open(file_path, "rb") as f:
                header_bytes = f.read(348)

        if len(header_bytes) < 348:
            return None

        # Endianness check using sizeof_hdr (must be 348)
        sizeof_hdr_le = struct.unpack("<i", header_bytes[0:4])[0]
        sizeof_hdr_be = struct.unpack(">i", header_bytes[0:4])[0]

        if sizeof_hdr_le == 348:
            endian = "<"
        elif sizeof_hdr_be == 348:
            endian = ">"
        else:
            return None

        # dim array: short dim[8] at offset 40
        dim = struct.unpack(f"{endian}8h", header_bytes[40:56])
        ndim = dim[0]
        if ndim < 3:
            return None

        shape = tuple(dim[1:ndim + 1])
        num_timepoints = dim[4] if ndim >= 4 else 1

        # pixdim array: float pixdim[8] at offset 76
        pixdim = struct.unpack(f"{endian}8f", header_bytes[76:108])
        voxel_dims = tuple(round(float(pixdim[i]), 4) for i in range(1, min(4, ndim + 1)))
        hdr_tr = round(float(pixdim[4]), 4) if ndim >= 4 else None

        return {
            "nifti_shape": str(shape),
            "num_timepoints": int(num_timepoints),
            "voxel_dimensions": str(voxel_dims),
            "header_tr_seconds": hdr_tr,
        }
    except Exception as e:
        return None


def get_possible_local_paths(site_code: str, participant_id: str) -> list:
    """
    Generates potential local file paths for raw BOLD scan based on ABIDE II BIDS structure.
    """
    site_clean = str(site_code).strip().replace("ABIDEII-", "")
    sub_clean = str(participant_id).strip().replace("sub-", "")
    site_bids = f"ABIDEII-{site_clean}"
    sub_bids = f"sub-{sub_clean}"

    return [
        RAW_FMRI_DIR / site_bids / sub_bids / "ses-1" / "func" / f"{sub_bids}_ses-1_task-rest_run-1_bold.nii.gz",
        RAW_FMRI_DIR / site_bids / sub_bids / "func" / f"{sub_bids}_task-rest_run-1_bold.nii.gz",
        PROJECT_ROOT / "data" / "raw" / site_bids / sub_bids / "ses-1" / "func" / f"{sub_bids}_ses-1_task-rest_run-1_bold.nii.gz",
        RAW_FMRI_DIR / sub_bids / f"{sub_bids}_task-rest_run-1_bold.nii.gz",
        RAW_FMRI_DIR / f"{sub_bids}_bold.nii.gz",
    ]


def main():
    print("=" * 70)
    print("STEP 3: RS-FMRI BOLD DATA MANIFEST CREATION & COHORT VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. LOAD VERIFIED PERSON 2 SEVERITY COHORT
    # -------------------------------------------------------------------------
    print(f"\n[1] Loading verified cohort: {COHORT_PATH}")
    if not COHORT_PATH.exists():
        raise FileNotFoundError(f"Cohort CSV missing at {COHORT_PATH}!")

    # Load IDs explicitly as strings to preserve lead zeros or formatting
    df_cohort = pd.read_csv(COHORT_PATH, dtype={"participant_id": str})
    df_cohort["participant_id"] = df_cohort["participant_id"].str.strip()

    total_cohort_size = len(df_cohort)
    print(f"Total subjects loaded: {total_cohort_size}")

    # -------------------------------------------------------------------------
    # 2. VERIFY COHORT CONSTRAINTS
    # -------------------------------------------------------------------------
    print("\n[2] Verifying cohort constraints...")

    # Check N = 228
    assert total_cohort_size == 228, f"EXPECTED 228 subjects, but found {total_cohort_size}!"

    # Check for duplicate participant IDs
    num_duplicates = df_cohort["participant_id"].duplicated().sum()
    assert num_duplicates == 0, f"Found {num_duplicates} duplicate participant IDs in cohort!"

    # Verify severity counts
    severity_counts = df_cohort["severity_class"].value_counts().to_dict()
    exp_severity = {"Low": 37, "Moderate": 87, "High": 104}
    for sev_cat, exp_val in exp_severity.items():
        actual_val = severity_counts.get(sev_cat, 0)
        assert actual_val == exp_val, f"Severity '{sev_cat}' mismatch: Expected {exp_val}, got {actual_val}"

    # Verify split counts
    split_counts = df_cohort["split"].value_counts().to_dict()
    exp_splits = {"train": 149, "val": 37, "test": 42}
    for split_cat, exp_val in exp_splits.items():
        actual_val = split_counts.get(split_cat, 0)
        assert actual_val == exp_val, f"Split '{split_cat}' mismatch: Expected {exp_val}, got {actual_val}"

    print("[OK] Cohort Verification PASSED!")
    print(f"   - Total Subjects: {total_cohort_size}")
    print(f"   - Severity Class: Low={severity_counts['Low']}, Moderate={severity_counts['Moderate']}, High={severity_counts['High']}")
    print(f"   - Split Assignment: Train={split_counts['train']}, Validation={split_counts['val']}, Test={split_counts['test']}")
    print("   - Duplicate Participant IDs: 0")

    # -------------------------------------------------------------------------
    # 3. BUILD MANIFEST ENTRIES
    # -------------------------------------------------------------------------
    print("\n[3] Generating BOLD file manifest and checking local availability...")

    manifest_rows = []
    bold_found_count = 0
    bold_missing_count = 0
    nifti_inspected = False

    for idx, row in df_cohort.iterrows():
        sub_id = str(row["participant_id"])
        site_code = str(row["site_code"])
        split = str(row["split"])
        sev_class = str(row["severity_class"])
        ados_score = row["ados_2_severity_total"]

        # 1. Site TR
        try:
            site_tr = sdu.get_site_tr(site_code)
        except Exception:
            site_tr = None

        # 2. S3 URI
        s3_uri = sdu.get_s3_func_path(site_code, sub_id)

        # 3. Local File Availability Check
        candidate_paths = get_possible_local_paths(site_code, sub_id)
        primary_expected_path = candidate_paths[0]

        file_found = False
        actual_path = primary_expected_path
        nifti_meta = None

        for cand_path in candidate_paths:
            if cand_path.exists():
                file_found = True
                actual_path = cand_path
                nifti_meta = parse_nifti_header(actual_path)
                nifti_inspected = True
                break

        if file_found:
            bold_found_count += 1
        else:
            bold_missing_count += 1

        # Relative expected path for clean output
        try:
            rel_expected_path = str(actual_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_expected_path = str(actual_path)

        manifest_entry = {
            "participant_id": sub_id,
            "site_code": site_code,
            "split": split,
            "severity_class": sev_class,
            "ados_2_severity_total": ados_score,
            "expected_bold_path": rel_expected_path,
            "s3_bold_uri": s3_uri,
            "bold_file_exists": file_found,
            "tr_seconds": site_tr,
            "nifti_shape": nifti_meta["nifti_shape"] if nifti_meta else "",
            "num_timepoints": nifti_meta["num_timepoints"] if nifti_meta else "",
            "voxel_dimensions": nifti_meta["voxel_dimensions"] if nifti_meta else "",
        }
        manifest_rows.append(manifest_entry)

    df_manifest = pd.DataFrame(manifest_rows)

    # -------------------------------------------------------------------------
    # 4. SAVE MANIFEST
    # -------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(MANIFEST_OUTPUT_PATH, index=False)
    print(f"\n[4] Saved BOLD manifest to: {MANIFEST_OUTPUT_PATH}")

    # -------------------------------------------------------------------------
    # 5. DISPLAY DETAILED AUDIT REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BOLD DATA AUDIT REPORT")
    print("=" * 70)
    print(f"Total Cohort Subjects: {len(df_manifest)}")
    print(f"Local BOLD Files Found:   {bold_found_count}")
    print(f"Local BOLD Files Missing: {bold_missing_count}")

    print("\n--- BOLD Availability by Split ---")
    split_avail = pd.crosstab(df_manifest["split"], df_manifest["bold_file_exists"])
    split_avail = split_avail.reindex(["train", "val", "test"])
    print(split_avail.to_string())

    print("\n--- BOLD Availability by Severity Class ---")
    sev_avail = pd.crosstab(df_manifest["severity_class"], df_manifest["bold_file_exists"])
    sev_avail = sev_avail.reindex(["Low", "Moderate", "High"])
    print(sev_avail.to_string())

    print("\n--- Sample Expected BOLD Paths (First 5) ---")
    for sample_row in manifest_rows[:5]:
        print(f"Sub: {sample_row['participant_id']} | Site: {sample_row['site_code']} | Local: {sample_row['expected_bold_path']} | Exists: {sample_row['bold_file_exists']}")
        print(f"     S3: {sample_row['s3_bold_uri']}")

    if bold_missing_count > 0:
        print("\nNOTE: Local raw rs-fMRI (BOLD) files are not present in data/raw/.")
        print("Per Step 3 instructions, automatic downloading was skipped.")
        print("The manifest is fully prepared with expected local paths and S3 URIs for downloading/preprocessing in subsequent steps.")

    # -------------------------------------------------------------------------
    # 6. CONCISE STATUS SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3 STATUS")
    print("=" * 70)
    print("Cohort verified: YES")
    print("Manifest created: YES")
    print(f"Subjects in manifest: {len(df_manifest)}")
    print(f"BOLD files found: {bold_found_count}")
    print(f"BOLD files missing: {bold_missing_count}")
    print(f"NIfTI inspection performed: {'YES' if nifti_inspected else 'NO'}")
    print("Preprocessing performed: NO")
    print("=" * 70)


if __name__ == "__main__":
    main()
