#!/usr/bin/env python3
"""
src/shared_data_utils.py
Shared Dataset & Data Loading Utilities for ABIDE II Multimodal Project.

Provides standardized helper functions for Person 1 (T1 3D ResNet) and Person 2 (rs-fMRI GAT)
to load master subject lists, train/val/test splits, site scanner metadata, and format S3 paths.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PHENO_DIR = DATA_DIR / "phenotypic"
SPLITS_DIR = PHENO_DIR / "splits"

MASTER_MANIFEST_PATH = PHENO_DIR / "split_master_manifest.csv"
SITE_SCANNER_REF_PATH = PHENO_DIR / "site_scanner_reference.csv"
ABIDE_PHENO_PATH = PHENO_DIR / "abide2_phenotypic.csv"
FINAL_SUBJECT_LIST_PATH = PHENO_DIR / "final_subject_list.csv"

S3_BUCKET = "fcp-indi"
S3_BASE_PREFIX = "data/Projects/ABIDE2/RawData"


def load_master_manifest(manifest_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the canonical split_master_manifest.csv containing all 1,007 split participants,
    their demographic metadata, train/val/test split tags, and modality availability flags.
    """
    path = manifest_path or MASTER_MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(f"Master manifest missing at {path}. Run data preparation stage first!")
    df = pd.read_csv(path, dtype=str)
    return df


def load_split(
    split_name: str,
    cohort: str = "all",
    manifest_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Loads a specific split ('train', 'val', or 'test') filtered by cohort.

    Args:
        split_name: One of 'train', 'val', 'test'.
        cohort: 
            - 'all': All 1,007 split participants (single-modality structural T1).
            - 'dual_modality': Filtered to subjects with both raw T1 and rs-fMRI scans available on S3 (975 subjects).
            - 'preprocessed_t1': Filtered to subjects with completed T1 preprocessed .npy files.

    Returns:
        pd.DataFrame containing subject records for the requested split.
    """
    split_name = split_name.lower().strip()
    if split_name not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split_name '{split_name}'. Must be one of ['train', 'val', 'test'].")

    df_master = load_master_manifest(manifest_path)
    df_split = df_master[df_master["split"] == split_name].copy()

    if cohort == "dual_modality":
        df_split = df_split[df_split["has_dual_modality"].astype(str).str.upper() == "TRUE"].copy()
    elif cohort == "preprocessed_t1":
        df_split = df_split[df_split["has_t1_preprocessed"].astype(str).str.upper() == "TRUE"].copy()
    elif cohort != "all":
        raise ValueError(f"Invalid cohort '{cohort}'. Must be one of ['all', 'dual_modality', 'preprocessed_t1'].")

    return df_split.reset_index(drop=True)


def load_site_scanner_info(ref_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the site scanner reference table containing repetition time (TR), manufacturer,
    scanner model, and recommended bandpass filtering limits for all 16 ABIDE II sites.
    """
    path = ref_path or SITE_SCANNER_REF_PATH
    if not path.exists():
        raise FileNotFoundError(f"Site scanner reference table missing at {path}.")
    df = pd.read_csv(path)
    return df


def get_site_tr(site_code: str, ref_path: Optional[Path] = None) -> float:
    """
    Returns the repetition time (TR in seconds) for a given site_code.
    """
    df_info = load_site_scanner_info(ref_path)
    clean_site = str(site_code).strip().replace("ABIDEII-", "")
    match = df_info[df_info["site_code"] == clean_site]
    if len(match) == 0:
        raise KeyError(f"Site code '{site_code}' not found in site scanner reference table.")
    return float(match.iloc[0]["tr_seconds"])


def get_s3_func_path(site_code: str, participant_id: str, run: int = 1) -> str:
    """
    Formats the expected S3 URI for a subject's raw rs-fMRI BOLD scan.
    Example: s3://fcp-indi/data/Projects/ABIDE2/RawData/ABIDEII-USM_1/sub-29495/ses-1/func/sub-29495_ses-1_task-rest_run-1_bold.nii.gz
    """
    site_clean = str(site_code).strip().replace("ABIDEII-", "")
    sub_clean = str(participant_id).strip().replace("sub-", "")
    site_raw = f"ABIDEII-{site_clean}"
    return (
        f"s3://{S3_BUCKET}/{S3_BASE_PREFIX}/{site_raw}/sub-{sub_clean}/ses-1/func/"
        f"sub-{sub_clean}_ses-1_task-rest_run-{run}_bold.nii.gz"
    )


def get_s3_t1_path(site_code: str, participant_id: str, run: int = 1) -> str:
    """
    Formats the expected S3 URI for a subject's raw T1 anatomical scan.
    Example: s3://fcp-indi/data/Projects/ABIDE2/RawData/ABIDEII-USM_1/sub-29495/ses-1/anat/sub-29495_ses-1_run-1_T1w.nii.gz
    """
    site_clean = str(site_code).strip().replace("ABIDEII-", "")
    sub_clean = str(participant_id).strip().replace("sub-", "")
    site_raw = f"ABIDEII-{site_clean}"
    return (
        f"s3://{S3_BUCKET}/{S3_BASE_PREFIX}/{site_raw}/sub-{sub_clean}/ses-1/anat/"
        f"sub-{sub_clean}_ses-1_run-{run}_T1w.nii.gz"
    )


def validate_split_consistency() -> bool:
    """
    Validates that train, val, and test splits are mutually exclusive and sum to 1,007 subjects.
    Returns True if validation passes, raises AssertionError otherwise.
    """
    df_master = load_master_manifest()
    train_subs = set(df_master[df_master["split"] == "train"]["participant_id"] + "_" + df_master[df_master["split"] == "train"]["site_code"])
    val_subs = set(df_master[df_master["split"] == "val"]["participant_id"] + "_" + df_master[df_master["split"] == "val"]["site_code"])
    test_subs = set(df_master[df_master["split"] == "test"]["participant_id"] + "_" + df_master[df_master["split"] == "test"]["site_code"])

    assert len(train_subs & val_subs) == 0, f"Train and Val overlap: {train_subs & val_subs}"
    assert len(train_subs & test_subs) == 0, f"Train and Test overlap: {train_subs & test_subs}"
    assert len(val_subs & test_subs) == 0, f"Val and Test overlap: {val_subs & test_subs}"
    assert len(df_master) == 1007, f"Expected 1,007 subjects, got {len(df_master)}"

    print(f"Validation Passed! Splits are mutually exclusive (Train: {len(train_subs)}, Val: {len(val_subs)}, Test: {len(test_subs)}, Total: {len(df_master)}).")
    return True


if __name__ == "__main__":
    validate_split_consistency()
