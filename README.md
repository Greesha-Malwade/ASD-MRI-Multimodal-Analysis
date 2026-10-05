# Multimodal MRI Analysis for Research-Defined ASD Severity Classification

[![Dataset](https://img.shields.io/badge/Dataset-ABIDE%20II%20RawData-blue)](https://fcp-indi.s3.amazonaws.com/index.html#data/Projects/ABIDE2/RawData/)
[![Pipeline](https://img.shields.io/badge/Pipeline-3D%20ResNet%20%2B%20rs--fMRI%20GAT-green)](#)
[![Status](https://img.shields.io/badge/Contract%20Status-Verified-success)](docs/shared_dataset_contract.md)

This repository contains the complete codebase, data manifests, preprocessing specifications, and developer hand-off contracts for the **ABIDE II Multimodal Autism Spectrum Disorder (ASD) Classification Project**. 

The project pairs a **3D ResNet-18 Structural T1 MRI Pipeline** (Person 1) with a **Graph Attention Network (GAT) Resting-State Functional MRI (rs-fMRI) Pipeline** (Person 2) to build a unified multimodal fusion classifier.

---

## 📌 Index of 10 Shared Contract Items & File Locations

Below is the definitive reference map pointing to where each of the 10 core dataset & preprocessing contract items is explained, documented, and stored in this repository:

| # | Contract Item | Primary Explanation & Documentation Location | Exact Data File / Manifest Location | Python API & Script Reference |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Dataset Version** | [`docs/shared_dataset_contract.md#1-dataset-version`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#1-dataset-version) | `s3://fcp-indi/data/Projects/ABIDE2/RawData/` | [`src/dataset_prep.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/dataset_prep.py), [`src/preprocess_t1.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/preprocess_t1.py) |
| **2** | **Final Subject List** | [`docs/shared_dataset_contract.md#2-final-subject-list`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#2-final-subject-list) | [`data/phenotypic/split_master_manifest.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/split_master_manifest.csv), [`data/phenotypic/final_subject_list.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/final_subject_list.csv) | `src.shared_data_utils.load_master_manifest()` |
| **3** | **Master CSV / Metadata** | [`docs/shared_dataset_contract.md#3-master-csv--metadata-file`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#3-master-csv--metadata-file) | [`data/phenotypic/abide2_phenotypic.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/abide2_phenotypic.csv) | `src.shared_data_utils.load_master_manifest()` |
| **4** | **ADOS-2 Severity Mapping** | [`docs/shared_dataset_contract.md#4-ados-2-severity-mapping`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#4-ados-2-severity-mapping), [`docs/person2_fmri_gat_guide.md`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/person2_fmri_gat_guide.md) | `ados_2_severity_total` column in [`data/phenotypic/abide2_phenotypic.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/abide2_phenotypic.csv) | [`src/severity_regression.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/severity_regression.py) |
| **5** | **Train / Val / Test Split** | [`docs/shared_dataset_contract.md#5-train--val--test-split`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#5-train--val--test-split) | [`data/phenotypic/splits/train_asd_tc.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/train_asd_tc.csv), [`val_asd_tc.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/val_asd_tc.csv), [`test_asd_tc.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/test_asd_tc.csv) | `src.shared_data_utils.load_split(split_name)` |
| **6** | **MRI Preprocessing Conventions** | [`docs/shared_dataset_contract.md#6-mri-preprocessing-conventions`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#6-mri-preprocessing-conventions), [`docs/person2_fmri_gat_guide.md#3-rs-fmri-preprocessing-contract`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/person2_fmri_gat_guide.md#3-rs-fmri-preprocessing-contract) | Output array directory: [`data/preprocessed/structural/`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/preprocessed/structural/) | [`src/preprocess_t1.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/preprocess_t1.py) |
| **7** | **Brain Atlas / Parcellation** | [`docs/shared_dataset_contract.md#7-brain-atlas--parcellation`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#7-brain-atlas--parcellation), [`docs/person2_fmri_gat_guide.md#4-functional-connectivity-graph-construction`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/person2_fmri_gat_guide.md#4-functional-connectivity-graph-construction) | Feature Matrix: [`data/features/structural_features.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/features/structural_features.csv) | [`src/extract_features.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/extract_features.py) |
| **8** | **Dual-Modality Coverage** | [`docs/shared_dataset_contract.md#8-dual-modality-subject-coverage`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#8-dual-modality-subject-coverage) | `has_dual_modality` column in [`data/phenotypic/split_master_manifest.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/split_master_manifest.csv) | `src.shared_data_utils.load_split(cohort='dual_modality')` |
| **9** | **Site / Scanner Metadata** | [`docs/shared_dataset_contract.md#9-site--scanner-metadata`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#9-site--scanner-metadata), [`docs/person2_fmri_gat_guide.md#3-rs-fmri-preprocessing-contract`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/person2_fmri_gat_guide.md#3-rs-fmri-preprocessing-contract) | [`data/phenotypic/site_scanner_reference.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/site_scanner_reference.csv) | `src.shared_data_utils.load_site_scanner_info()`, `get_site_tr()` |
| **10**| **ResNet Embedding Output Spec** | [`docs/shared_dataset_contract.md#10-structural-embedding-output-spec`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/shared_dataset_contract.md#10-structural-embedding-output-spec), [`docs/person2_fmri_gat_guide.md#6-graph-attention-network-gat-architecture--embedding-spec`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/person2_fmri_gat_guide.md#6-graph-attention-network-gat-architecture--embedding-spec) | Layer `self.avgpool` & `torch.flatten(x, 1)` outputting shape **`(B, 512)`** | [`src/models/resnet3d.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/models/resnet3d.py#L66-L67), [`src/train_resnet3d.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/train_resnet3d.py) |

---

## 🔍 Detailed Breakdown of the 10 Core Contract Items

### 1. Dataset Version & Provenance
- **Dataset**: ABIDE II (Autism Brain Imaging Data Exchange II), RawData.
- **Source Bucket**: `s3://fcp-indi/data/Projects/ABIDE2/RawData/`
- **Data Format**: **Raw BIDS Data** (not preprocessed derivative releases).
- **Sites Covered**: 16 active sites (`USM_1`, `UCLA_1`, `UCD_1`, `TCD_1`, `SDSU_1`, `ONRC_2`, `OHSU_1`, `NYU_2`, `NYU_1`, `KUL_3`, `KKI_1`, `IU_1`, `IP_1`, `GU_1`, `EMC_1`, `BNI_1`).
- **Ingestion Code**: [`src/dataset_prep.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/dataset_prep.py) and [`src/preprocess_t1.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/preprocess_t1.py).

### 2. Final Subject List & Master Manifest
- **Total Cohort Size**: **1,007 split subjects** (516 ASD, 491 Control/TC).
- **Master Manifest**: [`data/phenotypic/split_master_manifest.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/split_master_manifest.csv)
- **Columns**: `participant_id`, `site_code`, `dx_group`, `split`, `age_at_scan`, `sex`, `fiq`, `has_t1_preprocessed`, `has_func_available`, `has_t1_raw_available`, `has_dual_modality`.

### 3. Master CSV / Metadata File
- **Canonical Phenotypic Manifest**: [`data/phenotypic/abide2_phenotypic.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/abide2_phenotypic.csv) (1,007 rows × 147 columns).
- **Usable Demographic Fields**: `age_at_scan` (100%), `sex` (99.9%), `eye_status_at_scan` (99.9%), `fiq` (93.7%), `viq` (74.2%), `piq` (73.3%), `handedness_category` (69.2%).
- **Sparse Fields Warning**: Subscore batteries (`adi_r_*`, `cprs_*`, `casi_*`, `csi_*`, `rbsr_*`) suffer from >57%–98% NaN missingness and should not be used as global inputs.

### 4. ADOS-2 CSS → 3-Class Severity Mapping
- **Clinical Rule**: ADOS-2 Comparison Severity Scores (CSS, 1–10 scale) map to 3 discrete severity classes:
  - **Low Severity**: CSS 1 to 5 (n = 37 subjects)
  - **Moderate Severity**: CSS 6 to 7 (n = 87 subjects)
  - **High Severity**: CSS 8 to 10 (n = 104 subjects)
- **Contract Status**: In Person 1's code ([`src/severity_regression.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/severity_regression.py)), ADOS-2 severity is treated as a continuous regression task on `ados_2_severity_total` (Mean = 7.09 ± 1.88, Range = 1–10). Converting to 3-class classification requires joint confirmation before building Person 2's severity classifier.

### 5. Train / Validation / Test Subject Split
- **Methodology**: Subject-ID based split (70% Train / 15% Val / 15% Test) stratified by `site_code` using `random_state=42`.
- **Split CSVs**:
  - Train (704 subjects): [`data/phenotypic/splits/train_asd_tc.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/train_asd_tc.csv)
  - Validation (151 subjects): [`data/phenotypic/splits/val_asd_tc.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/val_asd_tc.csv)
  - Test (152 subjects): [`data/phenotypic/splits/test_asd_tc.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/test_asd_tc.csv)
- **Python API**: Load directly via `src.shared_data_utils.load_split("train")`.

### 6. MRI Preprocessing Conventions
- **Structural T1 Pipeline** ([`src/preprocess_t1.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/preprocess_t1.py)): N4 Bias Field Correction $\rightarrow$ ANTsPyNet deep learning skull-stripping $\rightarrow$ Affine registration to MNI152 template $\rightarrow$ Resampling to `(128, 128, 128)` grid $\rightarrow$ Brain voxel Z-score normalization.
- **rs-fMRI Requirements**: Person 2 performs 24-motion parameter + WM/CSF signal regression, motion scrubbing (FD > 0.5mm), and TR-dependent bandpass filtering. **Both modalities must register to the identical MNI152 template space**.

### 7. Brain Atlas / Parcellation
- **Structural Atlas**: Harvard-Oxford Cortical (`cort-maxprob-thr25-1mm`) and Subcortical (`sub-maxprob-thr25-1mm`) MaxProb atlases ([`src/extract_features.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/extract_features.py)).
- **Functional Atlas**: Person 2 uses AAL-116, Schaefer-100, or Harvard-Oxford. Aligning to Harvard-Oxford is recommended if cross-modal region-level explainability is required.

### 8. Dual-Modality Subject Coverage
- **Total Split Subjects**: 1,007 subjects
- **Raw T1 Scans on S3**: 1,007 subjects (100%)
- **Raw rs-fMRI Scans on S3**: 975 subjects (96.8%)
- **Dual-Modality Usable Sample Size**: **975 subjects** (31 subjects at `GU_1` and 1 at `IP_1` lack rs-fMRI).

### 9. Site / Scanner Metadata
- **Reference File**: [`data/phenotypic/site_scanner_reference.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/site_scanner_reference.csv)
- **TR Spectrum**: Varies from **0.475s** (multiband at `ONRC_2`) to **3.000s** (`BNI_1` and `UCLA_1`).
- **Nyquist Equation**: Person 2 must calculate $f_{Nyquist} = \frac{1}{2 \times TR}$ per site for temporal filtering.

### 10. ResNet Structural Embedding Output Spec
- **Model Definition**: 3D ResNet-18 in [`src/models/resnet3d.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/models/resnet3d.py).
- **Embedding Extraction Layer**: Post-`avgpool` (`AdaptiveAvgPool3d((1, 1, 1))`) and `torch.flatten(x, 1)`.
- **Output Tensor Shape**: **`(B, 512)`** (512-dimensional vector per subject).
- **GAT Integration**: Person 2's GAT model must include a projection layer `nn.Linear(d_gat, 512)` to produce a matching `(B, 512)` functional embedding vector for multimodal fusion.

---

## 💻 Quick Usage & Verification

### Running the Python Shared API
```python
import src.shared_data_utils as sdu

# Load Master Manifest
df_master = sdu.load_master_manifest()
print(f"Total Master Manifest Subjects: {len(df_master)}")

# Load Dual-Modality Train Split
train_dual = sdu.load_split("train", cohort="dual_modality")
print(f"Dual-Modality Train Subjects: {len(train_dual)}")

# Get Site TR
tr_usm = sdu.get_site_tr("USM_1")
print(f"USM_1 Repetition Time: {tr_usm} seconds")
```

### Verification Command
Run the verification suite to ensure all manifests, CSVs, and API functions operate cleanly:
```bash
python "C:\Users\ridhi malawade\.gemini\antigravity-ide\brain\87303d86-1a5f-4a4b-9575-c13570953c3b\scratch\verify_handoff_package.py"
```

---

## 📁 Repository Directory Structure

```text
mri 2/
├── README.md                           <- Comprehensive Repository Index & Contract Reference
├── docs/
│   ├── shared_dataset_contract.md      <- Canonical Shared Preprocessing & Dataset Contract
│   └── person2_fmri_gat_guide.md       <- Developer Hand-off Guide for Person 2 (rs-fMRI GAT)
├── data/
│   ├── phenotypic/
│   │   ├── abide2_phenotypic.csv       <- Cleaned Phenotypic Data (1,007 subjects)
│   │   ├── split_master_manifest.csv   <- Master Split & Modality Manifest
│   │   ├── site_scanner_reference.csv  <- Site Scanner & TR Reference Table
│   │   ├── final_subject_list.csv      <- Compact Master Subject List
│   │   └── splits/                     <- Stratified Split CSVs (train/val/test)
│   ├── preprocessed/
│   │   └── structural/                 <- Preprocessed T1 .npy Volumes (128x128x128)
│   └── features/
│       └── structural_features.csv     <- Harvard-Oxford Atlas Volumetric Feature Matrix
├── src/
│   ├── shared_data_utils.py            <- Shared Data Ingestion API for Person 1 & Person 2
│   ├── dataset_prep.py                 <- Stage 1: Phenotypic Download & Stratified Splits
│   ├── preprocess_t1.py                <- Stage 2: ANTsPy T1 Preprocessing Pipeline
│   ├── extract_features.py             <- Stage 3: Harvard-Oxford Atlas Feature Extraction
│   ├── baseline_models.py              <- Stage 4: Tabular ML Baselines (RF, XGBoost, SVM)
│   ├── train_resnet3d.py               <- Stage 5: 3D ResNet-18 Deep Learning Classifier
│   ├── severity_regression.py          <- Stage 6: ADOS-2 Severity Score Regression
│   ├── ensemble_models.py              <- Stage 7: Soft-Voting Ensemble (ResNet + XGBoost)
│   ├── gradcam_3d.py                   <- Stage 8: 3D Grad-CAM Explainability & Visual QC
│   └── models/
│       └── resnet3d.py                 <- PyTorch 3D ResNet-18 Architecture (512-dim embedding)
└── logs/                               <- Pipeline execution logs
```



