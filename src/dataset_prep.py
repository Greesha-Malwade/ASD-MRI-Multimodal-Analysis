#!/usr/bin/env python3
"""
src/dataset_prep.py
Stage 1 — Dataset preparation for ABIDE II T1 structural MRI pipeline.

- Loads/downloads data/phenotypic/abide2_phenotypic.csv
- Normalizes column names, renames 'site'/'site_id' to 'site_code'
- Reports population counts for dx_group, age_at_scan, sex, ados_2_severity_total
- Builds subject lists: (a) binary ASD/TC classification, (b) ASD severity classification
- Creates stratified train/val/test splits (70/15/15) by site_code
- Logs S3 T1 scan availability for subjects
"""

import logging
import os
import sys
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from sklearn.model_selection import train_test_split

SITES = [
    "USM_1", "UCLA_1", "UCD_1", "TCD_1", "SDSU_1", "ONRC_2",
    "OHSU_1", "NYU_2", "NYU_1", "KUL_3", "KKI_1", "IU_1",
    "IP_1", "GU_1", "ETHZ_1", "EMC_1", "BNI_1"
]

S3_BUCKET = "fcp-indi"
S3_BASE_PREFIX = "data/Projects/ABIDE2/RawData"


def setup_directories(project_root: Path):
    dirs = [
        project_root / "data" / "phenotypic" / "splits",
        project_root / "data" / "raw" / "abide2_t1",
        project_root / "data" / "preprocessed" / "structural",
        project_root / "data" / "features",
        project_root / "models" / "checkpoints",
        project_root / "results" / "gradcam",
        project_root / "src",
        project_root / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def download_or_load_phenotypic(pheno_path: Path) -> pd.DataFrame:
    if pheno_path.exists():
        logging.info(f"Loading existing phenotypic CSV from {pheno_path}")
        df = pd.read_csv(pheno_path, dtype=str)
    else:
        logging.info(f"Phenotypic file not found at {pheno_path}. Downloading from S3 across 17 sites...")
        all_dfs = []
        for site in SITES:
            url = f"https://{S3_BUCKET}.s3.amazonaws.com/{S3_BASE_PREFIX}/ABIDEII-{site}/participants.tsv"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp:
                    site_df = pd.read_csv(resp, sep="\t", encoding="latin1", dtype=str)
                    site_df["raw_site_name"] = site
                    all_dfs.append(site_df)
                logging.info(f"Downloaded {len(site_df)} participant records for site {site}")
            except Exception as e:
                logging.warning(f"Could not download TSV for site {site}: {e}")

        if not all_dfs:
            raise RuntimeError("Failed to download phenotypic data for all sites.")

        df = pd.concat(all_dfs, ignore_index=True)
        pheno_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(pheno_path, index=False)
        logging.info(f"Saved merged phenotypic data ({len(df)} records) to {pheno_path}")

    # Normalize column names: lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()

    # Rename site/site_id to site_code
    if "site_id" in df.columns:
        df = df.rename(columns={"site_id": "site_code"})
    elif "site" in df.columns:
        df = df.rename(columns={"site": "site_code"})
    elif "site_code" not in df.columns:
        if "raw_site_name" in df.columns:
            df["site_code"] = df["raw_site_name"]
        else:
            df["site_code"] = "UNKNOWN"

    # Clean site_code formatting
    df["site_code"] = df["site_code"].astype(str).str.replace("ABIDEII-", "", regex=False).str.strip()

    return df


def report_population_counts(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("STAGE 1: POPULATION DEMOGRAPHICS REPORT")
    print("=" * 80)
    print(f"Total participants loaded: {len(df)}")

    # dx_group: 1 = ASD, 2 = Control (TC)
    if "dx_group" in df.columns:
        dx_counts = df["dx_group"].value_counts(dropna=False).to_dict()
        print(f"\nDiagnosis Groups (dx_group):")
        print(f"  - ASD (dx_group=1): {dx_counts.get('1', 0)}")
        print(f"  - Control/TC (dx_group=2): {dx_counts.get('2', 0)}")
        print(f"  - Missing/NaN: {dx_counts.get(np.nan, 0) + dx_counts.get(None, 0)}")

    # age_at_scan
    if "age_at_scan" in df.columns:
        age_num = pd.to_numeric(df["age_at_scan"], errors="coerce")
        print(f"\nAge at Scan (years):")
        print(f"  - Valid count: {age_num.notna().sum()} / {len(df)}")
        print(f"  - Mean ± SD: {age_num.mean():.2f} ± {age_num.std():.2f}")
        print(f"  - Range: [{age_num.min():.2f}, {age_num.max():.2f}]")

    # sex (1 = Male, 2 = Female)
    if "sex" in df.columns:
        sex_counts = df["sex"].value_counts(dropna=False).to_dict()
        print(f"\nSex Distribution:")
        print(f"  - Male (1): {sex_counts.get('1', 0)}")
        print(f"  - Female (2): {sex_counts.get('2', 0)}")

    # ados_2_severity_total (ASD only)
    if "ados_2_severity_total" in df.columns:
        severity_num = pd.to_numeric(df["ados_2_severity_total"], errors="coerce")
        print(f"\nADOS-2 Severity Total (ados_2_severity_total):")
        print(f"  - Total non-null records: {severity_num.notna().sum()}")
        if "dx_group" in df.columns:
            asd_mask = df["dx_group"] == "1"
            asd_severity = severity_num[asd_mask]
            tc_severity = severity_num[~asd_mask]
            print(f"  - ASD group valid severity count: {asd_severity.notna().sum()} (Expected for ASD)")
            print(f"  - Control group NaN count: {tc_severity.isna().sum()} (Expected NaN for controls)")
            if asd_severity.notna().sum() > 0:
                print(f"  - ASD Severity Score Distribution:")
                print(f"      Mean ± SD: {asd_severity.mean():.2f} ± {asd_severity.std():.2f}")
                print(f"      Min: {asd_severity.min():.1f}, Median: {asd_severity.median():.1f}, Max: {asd_severity.max():.1f}")
                val_counts = asd_severity.value_counts().sort_index()
                print("      Frequency breakdown:")
                for val, cnt in val_counts.items():
                    print(f"        Score {val:g}: {cnt} subjects")
    print("=" * 80 + "\n")


def check_s3_t1_scans(df: pd.DataFrame) -> dict:
    logging.info("Checking S3 T1 scan availability for participants across all sites...")
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3_client.get_paginator("list_objects_v2")

    # Collect available S3 keys per site
    s3_available_subjects = set()
    site_scan_counts = {}

    for site in SITES:
        prefix = f"{S3_BASE_PREFIX}/ABIDEII-{site}/"
        count = 0
        try:
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if "/anat/" in key and (key.endswith(".nii.gz") or key.endswith(".nii")):
                        # Extract sub id e.g. sub-29177 or 29177
                        parts = key.split("/")
                        for part in parts:
                            if part.startswith("sub-") or (part.isdigit() and len(part) >= 4):
                                sub_id = part.replace("sub-", "")
                                sub_key = f"{site}_{sub_id}"
                                s3_available_subjects.add(sub_key)
                                count += 1
            site_scan_counts[site] = count
            logging.info(f"S3 Site {site}: Found {count} T1 anatomical files")
        except Exception as e:
            logging.warning(f"Error querying S3 for site {site}: {e}")

    print("\n" + "=" * 80)
    print("STAGE 1: S3 T1 SCAN AVAILABILITY AUDIT")
    print("=" * 80)
    
    # Match against df
    matched_count = 0
    missing_count = 0
    
    if "participant_id" in df.columns:
        for idx, row in df.iterrows():
            sub_id = str(row["participant_id"]).strip().replace("sub-", "")
            site_code = str(row["site_code"]).strip().replace("ABIDEII-", "")
            sub_key = f"{site_code}_{sub_id}"
            
            # Check direct match or site prefix match
            if sub_key in s3_available_subjects or any(sub_id in s for s in s3_available_subjects if site_code in s):
                matched_count += 1
            else:
                missing_count += 1

        print(f"Total phenotypic subjects: {len(df)}")
        print(f"Subjects with verified T1 scan available on S3: {matched_count} ({matched_count/len(df)*100:.1f}%)")
        print(f"Subjects missing S3 T1 scan: {missing_count}")
    else:
        print("Column 'participant_id' not found to match S3 scans.")
        
    print("=" * 80 + "\n")
    return s3_available_subjects


def create_stratified_splits(df: pd.DataFrame, project_root: Path):
    splits_dir = project_root / "data" / "phenotypic" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cohort A: Binary ASD/TC classification (all subjects with dx_group in [1, 2])
    df_asd_tc = df[df["dx_group"].isin(["1", "2"])].copy()

    # Stratified Split 70/15/15 by site_code
    # Helper to split safely with stratification where possible
    def split_70_15_15(data_df: pd.DataFrame, strat_col: str, random_state: int = 42):
        # Count per category
        counts = data_df[strat_col].value_counts()
        rare_categories = counts[counts < 3].index.tolist()
        
        if rare_categories:
            logging.info(f"Note: Sites with < 3 samples ({rare_categories}) will use random splitting for those instances.")
        
        # Train (70%) vs Temp (30%)
        try:
            train_df, temp_df = train_test_split(
                data_df, test_size=0.30, random_state=random_state, stratify=data_df[strat_col]
            )
        except ValueError:
            train_df, temp_df = train_test_split(
                data_df, test_size=0.30, random_state=random_state
            )

        # Temp (30%) -> Val (15%) and Test (15%)
        try:
            val_df, test_df = train_test_split(
                temp_df, test_size=0.50, random_state=random_state, stratify=temp_df[strat_col]
            )
        except ValueError:
            val_df, test_df = train_test_split(
                temp_df, test_size=0.50, random_state=random_state
            )

        return train_df, val_df, test_df

    # Split Cohort A (ASD vs TC)
    train_asd_tc, val_asd_tc, test_asd_tc = split_70_15_15(df_asd_tc, strat_col="site_code", random_state=42)

    train_asd_tc.to_csv(splits_dir / "train_asd_tc.csv", index=False)
    val_asd_tc.to_csv(splits_dir / "val_asd_tc.csv", index=False)
    test_asd_tc.to_csv(splits_dir / "test_asd_tc.csv", index=False)

    # 2. Cohort B: ASD Severity (dx_group == 1 and non-null ados_2_severity_total)
    df_severity = df[
        (df["dx_group"] == "1") & 
        (pd.to_numeric(df["ados_2_severity_total"], errors="coerce").notna())
    ].copy()

    train_sev, val_sev, test_sev = split_70_15_15(df_severity, strat_col="site_code", random_state=42)

    train_sev.to_csv(splits_dir / "train_severity.csv", index=False)
    val_sev.to_csv(splits_dir / "val_severity.csv", index=False)
    test_sev.to_csv(splits_dir / "test_severity.csv", index=False)

    print("\n" + "=" * 80)
    print("STAGE 1: TRAIN / VAL / TEST SPLIT SIZES SUMMARY")
    print("=" * 80)
    print("Cohort A: Binary ASD vs TC Classification (70 / 15 / 15)")
    print(f"  - Total Subjects: {len(df_asd_tc)}")
    print(f"  - Train split:    {len(train_asd_tc)} subjects ({len(train_asd_tc)/len(df_asd_tc)*100:.1f}%)")
    print(f"  - Val split:      {len(val_asd_tc)} subjects ({len(val_asd_tc)/len(df_asd_tc)*100:.1f}%)")
    print(f"  - Test split:     {len(test_asd_tc)} subjects ({len(test_asd_tc)/len(df_asd_tc)*100:.1f}%)")
    print(f"  - ASD/TC ratio in Train: { (train_asd_tc['dx_group']=='1').sum() } ASD / { (train_asd_tc['dx_group']=='2').sum() } TC")
    print(f"  - ASD/TC ratio in Val:   { (val_asd_tc['dx_group']=='1').sum() } ASD / { (val_asd_tc['dx_group']=='2').sum() } TC")
    print(f"  - ASD/TC ratio in Test:  { (test_asd_tc['dx_group']=='1').sum() } ASD / { (test_asd_tc['dx_group']=='2').sum() } TC")

    print("\nCohort B: ASD Severity Classification / Regression (70 / 15 / 15)")
    print(f"  - Total ASD Subjects with Severity: {len(df_severity)}")
    print(f"  - Train split:    {len(train_sev)} subjects ({len(train_sev)/len(df_severity)*100:.1f}%)")
    print(f"  - Val split:      {len(val_sev)} subjects ({len(val_sev)/len(df_severity)*100:.1f}%)")
    print(f"  - Test split:     {len(test_sev)} subjects ({len(test_sev)/len(df_severity)*100:.1f}%)")
    print("=" * 80 + "\n")


def main():
    project_root = Path(__file__).resolve().parent.parent
    setup_directories(project_root)

    log_file = project_root / "logs" / "dataset_prep.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Starting Stage 1: Dataset Preparation...")

    pheno_path = project_root / "data" / "phenotypic" / "abide2_phenotypic.csv"
    df_pheno = download_or_load_phenotypic(pheno_path)

    report_population_counts(df_pheno)
    check_s3_t1_scans(df_pheno)
    create_stratified_splits(df_pheno, project_root)

    logging.info("Stage 1 dataset preparation completed successfully.")


if __name__ == "__main__":
    main()
