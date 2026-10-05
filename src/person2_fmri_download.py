#!/usr/bin/env python3
"""
src/person2_fmri_download.py

Stage 3.5 — rs-fMRI (BOLD) Data Acquisition & Verification for Person 2 severity cohort.

This script:
1. Loads data/fmri/person2_fmri_manifest.csv (228 ASD subjects).
2. Verifies public S3 HTTP URIs for all 228 required rs-fMRI scans on fcp-indi.s3.amazonaws.com.
3. Downloads only the single raw BOLD scan per subject to data/raw/abide2_fmri/.
4. Supports resuming/retrying: skips downloading files that already exist locally and are valid (>0 bytes).
5. Inspects NIfTI binary headers to extract shape, time points, voxel dimensions, and TR.
6. Updates data/fmri/person2_fmri_manifest.csv with complete file availability and metadata.
7. Outputs detailed breakdown by split, severity class, and lists any failed subject IDs.
"""

import gzip
import os
import struct
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
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

# File paths
MANIFEST_PATH = PROJECT_ROOT / "data" / "fmri" / "person2_fmri_manifest.csv"
RAW_FMRI_DIR = PROJECT_ROOT / "data" / "raw" / "abide2_fmri"
COHORT_PATH = PROJECT_ROOT / "data" / "phenotypic" / "person2_severity_fmri_cohort.csv"

S3_BASE_HTTP = "https://fcp-indi.s3.amazonaws.com"
S3_BUCKET = "fcp-indi"


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
    except Exception:
        return None


def resolve_s3_key(site_code: str, participant_id: str) -> str:
    """
    Resolves the exact S3 object key for a subject's raw rs-fMRI scan on fcp-indi S3 bucket.
    """
    site_clean = str(site_code).strip().replace("ABIDEII-", "")
    sub_clean = str(participant_id).strip().replace("sub-", "")

    candidate_keys = [
        f"data/Projects/ABIDE2/RawData/ABIDEII-{site_clean}/sub-{sub_clean}/ses-1/func/sub-{sub_clean}_ses-1_task-rest_run-1_bold.nii.gz",
        f"data/Projects/ABIDE2/RawData/ABIDEII-{site_clean}/sub-{sub_clean}/ses-1/func/sub-{sub_clean}_ses-1_task-rest_bold.nii.gz",
        f"data/Projects/ABIDE2/RawData/ABIDEII-{site_clean}/sub-{sub_clean}/func/sub-{sub_clean}_task-rest_run-1_bold.nii.gz",
        f"data/Projects/ABIDE2/RawData/ABIDEII-{site_clean}/sub-{sub_clean}/func/sub-{sub_clean}_task-rest_bold.nii.gz",
    ]

    for key in candidate_keys:
        url = f"{S3_BASE_HTTP}/{key}"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return key
        except Exception:
            continue

    # Fallback to S3 XML directory query if static keys differ (e.g. acq tags)
    prefix = f"data/Projects/ABIDE2/RawData/ABIDEII-{site_clean}/sub-{sub_clean}/"
    xml_url = f"{S3_BASE_HTTP}/?prefix={prefix}"
    try:
        req = urllib.request.urlopen(xml_url, timeout=10)
        root = ET.fromstring(req.read())
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys = [
            elem.text
            for elem in root.findall(".//s3:Key", ns)
            if "bold.nii.gz" in elem.text or ("func" in elem.text and elem.text.endswith(".nii.gz"))
        ]
        if keys:
            return keys[0]
    except Exception:
        pass

    return candidate_keys[0]


def download_file_with_resume(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """
    Downloads a file from url to dest_path with retry handling and non-empty validation.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
                block_size = 65536
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)

            if temp_path.exists() and temp_path.stat().st_size > 0:
                if dest_path.exists():
                    dest_path.unlink()
                temp_path.rename(dest_path)
                return True
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            if attempt < max_retries:
                time.sleep(2)
            else:
                print(f"   [ERROR] Download failed for {url} after {max_retries} attempts: {e}")
                return False

    return False


def main():
    print("=" * 75)
    print("STEP 3.5: RS-FMRI (BOLD) DATA ACQUISITION & METADATA EXTRACTION")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. LOAD MANIFEST
    # -------------------------------------------------------------------------
    print(f"\n[1] Loading manifest: {MANIFEST_PATH}")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest missing at {MANIFEST_PATH}! Run Step 3 first.")

    df_manifest = pd.read_csv(MANIFEST_PATH, dtype={"participant_id": str})
    df_manifest["participant_id"] = df_manifest["participant_id"].str.strip()

    total_required = len(df_manifest)
    print(f"Total subjects in manifest: {total_required}")
    assert total_required == 228, f"EXPECTED 228 subjects, got {total_required}!"

    # -------------------------------------------------------------------------
    # 2. ACQUIRE / DOWNLOAD RS-FMRI SCANS
    # -------------------------------------------------------------------------
    print("\n[2] Checking local availability & downloading missing scans from S3...")

    obtained_count = 0
    failed_count = 0
    failed_sub_ids = []

    updated_rows = []

    for idx, row in df_manifest.iterrows():
        sub_id = str(row["participant_id"])
        site_code = str(row["site_code"])
        split = str(row["split"])
        sev_class = str(row["severity_class"])
        ados_score = row["ados_2_severity_total"]

        # Get site TR
        try:
            site_tr = sdu.get_site_tr(site_code)
        except Exception:
            site_tr = None

        # Resolve S3 Key & URI
        s3_key = resolve_s3_key(site_code, sub_id)
        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        http_url = f"{S3_BASE_HTTP}/{s3_key}"

        # Target local file path
        bids_filename = Path(s3_key).name
        site_bids = f"ABIDEII-{site_code.replace('ABIDEII-', '')}"
        sub_bids = f"sub-{sub_id.replace('sub-', '')}"
        local_file_path = RAW_FMRI_DIR / site_bids / sub_bids / "ses-1" / "func" / bids_filename

        # Check existing file
        file_exists = False
        download_status = "PENDING"

        if local_file_path.exists() and local_file_path.stat().st_size > 0:
            file_exists = True
            download_status = "EXISTS"
            obtained_count += 1
        else:
            print(f"[{idx+1:03d}/{total_required:03d}] Downloading sub-{sub_id} ({site_code})...", end="", flush=True)
            success = download_file_with_resume(http_url, local_file_path)
            if success:
                file_exists = True
                download_status = "SUCCESS"
                obtained_count += 1
                size_mb = local_file_path.stat().st_size / (1024 * 1024)
                print(f" SUCCESS ({size_mb:.1f} MB)")
            else:
                download_status = "FAILED"
                failed_count += 1
                failed_sub_ids.append(sub_id)
                print(" FAILED")

        # Get file size
        file_size_bytes = local_file_path.stat().st_size if (file_exists and local_file_path.exists()) else 0

        # Inspect NIfTI binary header
        nifti_meta = parse_nifti_header(local_file_path) if file_exists else None

        rel_local_path = str(local_file_path.relative_to(PROJECT_ROOT)) if local_file_path.exists() else str(local_file_path)

        updated_row = {
            "participant_id": sub_id,
            "site_code": site_code,
            "split": split,
            "severity_class": sev_class,
            "ados_2_severity_total": ados_score,
            "expected_bold_path": rel_local_path,
            "s3_bold_uri": s3_uri,
            "bold_file_exists": file_exists,
            "file_size_bytes": file_size_bytes,
            "download_status": download_status,
            "tr_seconds": site_tr,
            "nifti_shape": nifti_meta["nifti_shape"] if nifti_meta else "",
            "num_timepoints": nifti_meta["num_timepoints"] if nifti_meta else "",
            "voxel_dimensions": nifti_meta["voxel_dimensions"] if nifti_meta else "",
        }
        updated_rows.append(updated_row)

    df_updated_manifest = pd.DataFrame(updated_rows)

    # -------------------------------------------------------------------------
    # 3. SAVE UPDATED MANIFEST
    # -------------------------------------------------------------------------
    df_updated_manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\n[3] Manifest updated with scan metadata at: {MANIFEST_PATH}")

    # -------------------------------------------------------------------------
    # 4. FINAL AUDIT & BREAKDOWN REPORTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 3.5 AUDIT & ACQUISITION REPORT")
    print("=" * 75)
    print(f"TOTAL REQUIRED:        {total_required}")
    print(f"SUCCESSFULLY OBTAINED: {obtained_count}")
    print(f"FAILED:                {failed_count}")
    print(f"MISSING:               {total_required - obtained_count}")

    print("\n--- Breakdown by Split ---")
    split_summary = pd.crosstab(df_updated_manifest["split"], df_updated_manifest["bold_file_exists"])
    split_summary = split_summary.reindex(["train", "val", "test"])
    print(split_summary.to_string())

    print("\n--- Breakdown by Severity Class ---")
    sev_summary = pd.crosstab(df_updated_manifest["severity_class"], df_updated_manifest["bold_file_exists"])
    sev_summary = sev_summary.reindex(["Low", "Moderate", "High"])
    print(sev_summary.to_string())

    print("\n--- Failed Participant IDs ---")
    if failed_sub_ids:
        print(", ".join(failed_sub_ids))
    else:
        print("None (0 failed subjects)")

    print("\n" + "=" * 75)
    print("STEP 3.5 STATUS")
    print("=" * 75)
    print(f"TOTAL REQUIRED: {total_required}")
    print(f"SUCCESSFULLY OBTAINED: {obtained_count}")
    print(f"FAILED: {failed_count}")
    print(f"MISSING: {total_required - obtained_count}")
    print("=" * 75)


if __name__ == "__main__":
    main()
