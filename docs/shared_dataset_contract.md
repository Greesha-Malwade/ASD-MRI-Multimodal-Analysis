# Shared Dataset & Preprocessing Contract

**Document Purpose**: Hand-off contract between Person 1 (T1 3D ResNet Structural MRI Pipeline) and Person 2 (rs-fMRI Graph Attention Network / GAT Pipeline) to enforce data consistency, eliminate data leakage, and establish cross-modal compatibility prior to multimodal fusion model development.

**Date Compiled**: 2026-09-05  
**Canonical Repository Root**: `mri 2/`  
**Master Subject Manifest**: [`data/phenotypic/split_master_manifest.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/split_master_manifest.csv)  
**Site Scanner Reference**: [`data/phenotypic/site_scanner_reference.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/site_scanner_reference.csv)  
**Shared Python Data API**: [`src/shared_data_utils.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/shared_data_utils.py)  
**Person 2 Developer Guide**: [`docs/person2_fmri_gat_guide.md`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/docs/person2_fmri_gat_guide.md)  
**Phenotypic Source**: [`data/phenotypic/abide2_phenotypic.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/abide2_phenotypic.csv)

---

## 1. DATASET VERSION

> [!IMPORTANT]  
> **Raw BIDS Data Notice**: This project strictly utilizes **Raw BIDS Data** directly from the ABIDE II S3 bucket, **NOT** preprocessed derivative releases (e.g., CPAC, DPARSF, NIAK, or NDMG). Both pipelines must perform preprocessing starting from raw NIfTI files to maintain full provenance and reproducibility.

- **Dataset Release**: ABIDE II (Autism Brain Imaging Data Exchange II), RawData.
- **S3 Bucket URI**: `s3://fcp-indi/data/Projects/ABIDE2/RawData/`
- **Configured Site List (17 sites total)**:
  `USM_1`, `UCLA_1`, `UCD_1`, `TCD_1`, `SDSU_1`, `ONRC_2`, `OHSU_1`, `NYU_2`, `NYU_1`, `KUL_3`, `KKI_1`, `IU_1`, `IP_1`, `GU_1`, `ETHZ_1`, `EMC_1`, `BNI_1`.
  *(Note: 16 of these 17 sites contain active participant records in the phenotypic manifest; `ETHZ_1` contains zero phenotypic records).*
- **Download & Ingestion Scripts**:
  - Phenotypic Manifest Ingestion: [`src/dataset_prep.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/dataset_prep.py) (downloads site-level `participants.tsv` files from `https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE2/RawData/ABIDEII-{site}/participants.tsv`, normalizes column names, and maps `site_id`/`site` to `site_code`).
  - Structural Scan Ingestion: [`src/preprocess_t1.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/preprocess_t1.py) (fetches raw T1 scans on-the-fly from `s3://fcp-indi/data/Projects/ABIDE2/RawData/ABIDEII-{site_code}/sub-{participant_id}/[ses-1/]/anat/`).
  - Shared Python Ingestion Helper: [`src/shared_data_utils.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/shared_data_utils.py).

---

## 2. FINAL SUBJECT LIST & MASTER MANIFESTS

> [!NOTE]  
> The master subject manifests have been generated at:
> - [`data/phenotypic/split_master_manifest.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/split_master_manifest.csv) (Contains demographic metadata, split assignment, and availability flags for all 1,007 split subjects).
> - [`data/phenotypic/final_subject_list.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/final_subject_list.csv) (Compact master subject list).
> Person 2 **MUST** filter all rs-fMRI processing to this exact list of subject IDs.

- **Total Cohort Size**: **1,007 participants** (516 ASD, 491 TC/Control).
- **Master Manifest Columns**:
  1. `participant_id`: BIDS subject ID string without `sub-` prefix (e.g., `29495`).
  2. `site_code`: Normalized site identifier (e.g., `USM_1`).
  3. `dx_group`: Diagnostic group classification (`1` = ASD, `2` = Control / TC).
  4. `split`: Split assignment (`train`, `val`, or `test`).
  5. `age_at_scan`, `sex`, `fiq`: Primary demographic covariates.
  6. `has_t1_preprocessed`: Boolean flag (`True`/`False`) indicating whether Person 1's T1 preprocessing pipeline has completed and written `.npy` outputs to [`data/preprocessed/structural/`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/preprocessed/structural/).
  7. `has_func_available`: Boolean flag (`True`/`False`) indicating whether a raw rs-fMRI (`func/` BOLD) scan is available on S3.
  8. `has_dual_modality`: Boolean flag (`True`/`False`) indicating availability of both raw T1 and rs-fMRI on S3.

### Summary Audit Statistics
- **Phenotypic Split Subjects**: 1,007 subjects
- **T1 Preprocessing Log Completed (`logs/t1_preprocessing_processed.log`)**: 5 subjects (`USM_1_29495`, `USM_1_29497`, `USM_1_29498`, `USM_1_29500`, `USM_1_29502`)
- **Raw T1 Available on S3 (`has_t1_s3_raw`)**: 1,007 subjects (100.0%)
- **Raw rs-fMRI Available on S3 (`has_func_available`)**: 975 subjects (96.8%)

---

## 3. MASTER CSV / METADATA FILE

- **Canonical File**: [`data/phenotypic/abide2_phenotypic.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/abide2_phenotypic.csv)
- **Structure**: 1,007 rows × 147 columns (column names converted to lower-case, stripped, and `site_id`/`site` renamed to `site_code`).

### Usable Columns (>50% Non-Null)
Person 2 can safely use the following demographic and global clinical variables for covariate analysis, stratification, or confound control:

| Column Name | Non-Null Count | Valid Percentage | Description / Data Range |
| :--- | :---: | :---: | :--- |
| `participant_id` | 1007 / 1007 | 100.0% | Participant BIDS ID |
| `site_code` | 1007 / 1007 | 100.0% | Site location identifier (16 sites) |
| `dx_group` | 1007 / 1007 | 100.0% | `1` = ASD, `2` = TC |
| `age_at_scan` | 1007 / 1007 | 100.0% | Age in years (Mean: 14.92 ± 9.38, Range: [5.22, 64.00]) |
| `raw_site_name` | 1007 / 1007 | 100.0% | Original site identifier |
| `sex` | 1006 / 1007 | 99.9% | `1` = Male (824), `2` = Female (182) |
| `eye_status_at_scan` | 1006 / 1007 | 99.9% | `1` = Open, `2` = Closed |
| `fiq` | 944 / 1007 | 93.7% | Full-Scale IQ (Mean: 109.1 ± 16.5) |
| `viq` | 747 / 1007 | 74.2% | Verbal IQ (Mean: 107.5 ± 17.2) |
| `piq` | 738 / 1007 | 73.3% | Performance IQ (Mean: 108.6 ± 16.1) |
| `handedness_category` | 697 / 1007 | 69.2% | Right / Left / Ambidextrous |
| `pdd_dsm_iv_tr` | 595 / 1007 | 59.1% | DSM-IV-TR subtype classification |

> [!WARNING]  
> **Sparse Metadata Warning**: Do **NOT** attempt to use subscore batteries such as ADI-R (`adi_r_*`, 4.8% valid), CPRS (`cprs_*`, 9.9% valid), CASI (`casi_*`, 1.2%–8.9% valid), CSI (`csi_*`, 1.7% valid), or RBSR (`rbsr_*`, 23.3%–42.2% valid). These columns suffer from extreme missingness (>57%–98% NaN) and will lead to severe sample truncation if included in pipeline models.

---

## 4. ADOS-2 SEVERITY MAPPING

> [!WARNING]  
> **UNRESOLVED — Needs Joint Decision Before Severity Classification Building**:  
> In Person 1's codebase ([`src/dataset_prep.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/dataset_prep.py) and [`src/severity_regression.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/severity_regression.py)), ADOS-2 severity is treated **exclusively as a continuous regression target** (`ados_2_severity_total` ranging from 1 to 10). **No discrete Low / Moderate / High classification cutoff rule has been finalized or coded anywhere in the project.**

### Empirical ADOS-2 Severity Score Distribution (`ados_2_severity_total`)
- **Total Valid Records**: 230 / 1007 (22.8% overall)
- **ASD Group Valid Records**: 228 / 516 ASD subjects (44.2% coverage; 246 ASD subjects are missing ADOS-2 scores)
- **TC Group Valid Records**: 2 / 491 Control subjects (Controls are expected NaN)
- **ASD Group Summary Statistics**:
  - **Mean ± SD**: 7.09 ± 1.88
  - **Min**: 1.0 | **25%**: 6.0 | **Median (50%)**: 7.0 | **75%**: 8.0 | **Max**: 10.0

```
  Frequency Breakdown of ADOS-2 Severity Scores (ASD Group):
  Score 1:  1 subject  [#]
  Score 2:  2 subjects [##]
  Score 3:  6 subjects [######]
  Score 4: 18 subjects [──────────────────]
  Score 5: 10 subjects [──────────]
  Score 6: 46 subjects [──────────────────────────────────────────────]
  Score 7: 41 subjects [─────────────────────────────────────────]
  Score 8: 50 subjects [──────────────────────────────────────────────────]
  Score 9: 32 subjects [────────────────────────────────────────]
  Score 10: 22 subjects [──────────────────────]
```

### Action Required Before Hand-off
Person 1 and Person 2 must decide between:
1. **Continuous Regression**: Predict exact `ados_2_severity_total` score (1–10) directly (matching [`src/severity_regression.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/severity_regression.py)).
2. **Binary Cutoff**: Low/Moderate Severity (1–6) vs. High Severity (7–10).
3. **Ternary Cutoff**: Low (1–5, n=37) vs. Moderate (6–7, n=87) vs. High (8–10, n=104).

---

## 5. TRAIN / VAL / TEST SPLIT

> [!IMPORTANT]  
> **FULLY RESOLVED — Identical Subject Split Enforcement**:  
> To prevent data leakage and ensure cross-modal evaluation validity, Person 2 **MUST NOT** generate a separate random split. Person 2 must load the exact CSV split files from [`data/phenotypic/splits/`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/) or use `src/shared_data_utils.load_split()`.

- **Split Protocol**: Subject-ID based split (70% Train / 15% Val / 15% Test) stratified by `site_code` using `random_state=42`.
- **Split CSV Files**:
  - [`data/phenotypic/splits/train_asd_tc.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/train_asd_tc.csv) (704 subjects)
  - [`data/phenotypic/splits/val_asd_tc.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/val_asd_tc.csv) (151 subjects)
  - [`data/phenotypic/splits/test_asd_tc.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/test_asd_tc.csv) (152 subjects)

### Cohort A: Binary ASD vs. Control Overall Counts
- **Train Set**: 704 subjects (354 ASD / 350 TC) — 69.9%
- **Val Set**: 151 subjects (76 ASD / 75 TC) — 15.0%
- **Test Set**: 152 subjects (76 ASD / 76 TC) — 15.1%
- **Total**: 1,007 subjects (516 ASD / 491 TC)

### Per-Site Subject Breakdown Across Splits

| Site Code | Train ASD | Train TC | Val ASD | Val TC | Test ASD | Test TC | Total Subjects |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BNI_1** | 20 | 21 | 5 | 4 | 4 | 4 | 58 |
| **EMC_1** | 19 | 19 | 4 | 4 | 4 | 4 | 54 |
| **GU_1** | 30 | 44 | 12 | 4 | 9 | 7 | 106 |
| **IP_1** | 15 | 24 | 3 | 5 | 4 | 5 | 56 |
| **IU_1** | 13 | 15 | 4 | 2 | 3 | 3 | 40 |
| **KKI_1** | 32 | 115 | 12 | 20 | 12 | 20 | 211 |
| **KUL_3** | 20 | 0 | 4 | 0 | 4 | 0 | 28 |
| **NYU_1** | 35 | 20 | 5 | 6 | 8 | 4 | 78 |
| **NYU_2** | 19 | 0 | 4 | 0 | 4 | 0 | 27 |
| **OHSU_1** | 22 | 43 | 5 | 9 | 10 | 4 | 93 |
| **ONRC_2** | 15 | 26 | 4 | 5 | 5 | 4 | 59 |
| **SDSU_1** | 22 | 19 | 4 | 4 | 7 | 2 | 58 |
| **TCD_1** | 17 | 12 | 2 | 5 | 2 | 4 | 42 |
| **UCD_1** | 12 | 10 | 3 | 2 | 3 | 2 | 32 |
| **UCLA_1** | 12 | 10 | 2 | 3 | 2 | 3 | 32 |
| **USM_1** | 12 | 11 | 3 | 2 | 2 | 3 | 33 |
| **TOTAL** | **354** | **350** | **76** | **75** | **76** | **76** | **1,007** |

---

## 6. MRI PREPROCESSING CONVENTIONS

### Structural (T1) Pipeline Parameters (Person 1 Ownership)
Documented from [`src/preprocess_t1.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/preprocess_t1.py):
1. **Bias Field Correction**: ANTsPy N4 Bias Field Correction (`ants.n4_bias_field_correction(raw_img)`).
2. **Skull-stripping**: ANTsPyNet deep learning brain extraction (`antspynet.brain_extraction(n4_img, modality="t1")`), thresholded with binary mask at probability 0.5.
3. **Spatial Normalization**: Affine registration to MNI152 standard template (`ants.registration(fixed=mni_template, moving=brain_img, type_of_transform="Affine")`).
4. **Target Resample Grid**: Volume shape `(128, 128, 128)` with linear interpolation (`ants.resample_image(..., (128, 128, 128), use_voxels=True, interp_type=1)`).
5. **Intensity Normalization**: Z-score normalization computed strictly over brain voxels (> 0.01 × max intensity). Preprocessed arrays saved as single-precision floating point `.npy` files.

### Functional (rs-fMRI) Scope & Shared Requirements
- **Person 2 Ownership**: Person 2 independently specifies rs-fMRI temporal processing parameters:
  - Bandpass filtering range (e.g., 0.01–0.1 Hz).
  - Confound regression (e.g., 24-parameter head motion + CSF/WM signals + global signal decision).
  - Motion scrub/framewise displacement (FD) thresholding (e.g., FD > 0.5mm).
- **SHARED REQUIREMENT**:
  Both modalities **MUST** register moving subject volumes to the **identical MNI152 template space** (1mm MNI152 coordinate space) so anatomical and functional regions align spatially.

---

## 7. BRAIN ATLAS / PARCELLATION

- **Structural Atlas Used (Person 1)**:
  Harvard-Oxford Cortical (`cort-maxprob-thr25-1mm`, 48 regions per hemisphere) and Subcortical (`sub-maxprob-thr25-1mm`, 21 regions) MaxProb atlases, resampled to the `(128, 128, 128)` grid ([`src/extract_features.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/extract_features.py)).

> [!WARNING]  
> **UNRESOLVED — FLAGGED FOR JOINT DECISION**:  
> Person 2's rs-fMRI GAT pipeline may use functional connectivity parcellations such as AAL-116, Schaefer-100/400, CC200, or Craddock-200.  
> **If we plan to perform structural-functional region-level comparison or fusion explainability (e.g., cross-modal attention maps), we MUST align atlas choices now before Person 2 constructs the GAT graph adjacency matrices.**  
> *Recommendation*: Standardize on either Harvard-Oxford or Schaefer-100 for both pipelines, or define an explicit mapping matrix ($M \in \mathbb{R}^{N_{struct} \times N_{func}}$) between region sets.

---

## 8. DUAL-MODALITY SUBJECT COVERAGE

Cross-referencing [`data/phenotypic/split_master_manifest.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/split_master_manifest.csv) against S3 BIDS directories:

- **Total Phenotypic Split Cohort**: 1,007 subjects
- **Raw T1 Anatomical Scans on S3**: 1,007 / 1,007 (100.0%)
- **Raw rs-fMRI Functional Scans on S3 (`has_func_available`)**: 975 / 1,007 (96.8%)
- **Dual-Modality Usable Sample Size**: **975 subjects**

> [!NOTE]  
> **32 Subjects Lack Functional Scans**: 32 subjects have valid T1 anatomical scans on S3 but lack rs-fMRI BOLD runs in S3 RawData. These consist of 31 subjects at site `GU_1` and 1 subject at site `IP_1`.  
> For single-modality structural experiments, all 1,007 subjects are usable. For multimodal fusion models, the effective dataset size is **975 subjects**.

---

## 9. SITE / SCANNER METADATA

Extracted directly from BIDS sidecar JSON metadata (`*task-rest_bold.json` / `*T1w.json`) on S3 and stored at [`data/phenotypic/site_scanner_reference.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/site_scanner_reference.csv):

| Site Code | Total Subjects | rs-fMRI Scans | Scanner Manufacturer | Scanner Model | Field Strength | Repetition Time (TR) | Bandpass Range (Hz) |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **BNI_1** | 58 | 58 | Philips | Ingenia | 3.0T | 3.000s | 0.01 – 0.1000 Hz |
| **EMC_1** | 54 | 54 | GE | Discovery MR750 | 3.0T | 2.000s | 0.01 – 0.1000 Hz |
| **GU_1** | 106 | 75 | Siemens | TrioTim | 3.0T | 2.000s | 0.01 – 0.1000 Hz |
| **IP_1** | 56 | 55 | Philips | Achieva | 3.0T | 2.700s | 0.01 – 0.1000 Hz |
| **IU_1** | 40 | 40 | Siemens | TrioTim (Multiband) | 3.0T | 0.813s | 0.01 – 0.1000 Hz |
| **KKI_1** | 211 | 211 | Philips | Achieva | 3.0T | 2.500s | 0.01 – 0.1000 Hz |
| **KUL_3** | 28 | 28 | Philips | Achieva dStream | 3.0T | 2.500s | 0.01 – 0.1000 Hz |
| **NYU_1** | 78 | 78 | Siemens | Allegra | 3.0T | 2.000s | 0.01 – 0.1000 Hz |
| **NYU_2** | 27 | 27 | Siemens | Allegra | 3.0T | 2.000s | 0.01 – 0.1000 Hz |
| **OHSU_1** | 93 | 93 | Siemens | TrioTim | 3.0T | 2.500s | 0.01 – 0.1000 Hz |
| **ONRC_2** | 59 | 59 | Siemens | Skyra (Multiband) | 3.0T | 0.475s | 0.01 – 0.1000 Hz |
| **SDSU_1** | 58 | 58 | GE | Discovery MR750 | 3.0T | 2.000s | 0.01 – 0.1000 Hz |
| **TCD_1** | 42 | 42 | Philips | Achieva | 3.0T | 2.000s | 0.01 – 0.1000 Hz |
| **UCD_1** | 32 | 32 | Siemens | TrioTim | 3.0T | 2.000s | 0.01 – 0.1000 Hz |
| **UCLA_1** | 32 | 32 | Siemens | TrioTim | 3.0T | 3.000s | 0.01 – 0.1000 Hz |
| **USM_1** | 33 | 33 | Siemens | TrioTim | 3.0T | 2.000s | 0.01 – 0.1000 Hz |

> [!CRITICAL]  
> **fMRI Filtering Dependency**: TR varies significantly by site, ranging from fast multiband sequences (TR = 0.475s at `ONRC_2`, TR = 0.813s at `IU_1`) to standard sequences (TR = 3.000s at `BNI_1` & `UCLA_1`). Person 2 **MUST** calculate site-specific Nyquist frequencies ($f_{Nyquist} = \frac{1}{2 \times TR}$) for temporal bandpass filtering.

---

## 10. STRUCTURAL EMBEDDING OUTPUT SPEC

- **Model Class**: 3D ResNet-18 (`ResNet3D` in [`src/models/resnet3d.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/models/resnet3d.py)).
- **Input Dimensions**: Single-channel 3D volume grid `(B, 1, 128, 128, 128)`.
- **Feature Extraction Layer**: Global Average Pooling layer output `self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))` followed by `torch.flatten(x, 1)`.
- **Planned Embedding Dimension**: **512-dimensional vector** (`(B, 512)` tensor).

### Multimodal Fusion Hand-off Specification for Person 2
To ensure seamless integration in the downstream fusion head (e.g. cross-attention or vector concatenation):
1. **Option A (Matching Dimension)**: Person 2 designs the GAT readout/pooling layer to output a **512-dimensional graph embedding** $h_{func} \in \mathbb{R}^{B \times 512}$.
2. **Option B (Projection Head)**: If Person 2's GAT outputs dimension $d_{gat}$ (e.g., $d_{gat} = 128$ or $256$), a linear projection layer `nn.Linear(d_gat, 512)` will be added in the fusion module prior to computing cross-modal attention maps.

---

## CONCLUDED CONTRACT SUMMARY & HAND-OFF CHECKLIST

| Item | Topic | Contract Status | Resolution Details / Action Item |
| :---: | :--- | :---: | :--- |
| **1** | Dataset Version | **FULLY RESOLVED** | ABIDE II Raw BIDS Data from `s3://fcp-indi/data/Projects/ABIDE2/RawData/` across 16 active sites. |
| **2** | Final Subject List | **FULLY RESOLVED** | 1,007 subjects saved to [`data/phenotypic/split_master_manifest.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/split_master_manifest.csv). |
| **3** | Master Phenotypic CSV | **FULLY RESOLVED** | Canonical manifest at [`data/phenotypic/abide2_phenotypic.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/abide2_phenotypic.csv); usable vs. sparse subscores documented. |
| **4** | ADOS-2 Severity Mapping | **FLAGGED / UNRESOLVED** | Code uses continuous regression; discrete Low/Mod/High cutoffs need joint decision. Distribution provided. |
| **5** | Train/Val/Test Split | **FULLY RESOLVED** | Identical 70/15/15 site-stratified split enforced via [`data/phenotypic/splits/`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/splits/) and [`src/shared_data_utils.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/shared_data_utils.py). |
| **6** | MRI Preprocessing | **FULLY RESOLVED** | Structural parameters fixed (N4, ANTsPyNet skullstrip, Affine MNI152, 128³ shape, Z-score). Both register to MNI152. |
| **7** | Atlas / Parcellation | **FLAGGED / UNRESOLVED** | Person 1 uses Harvard-Oxford; Person 2 GAT atlas needs joint agreement for region-level explainability. |
| **8** | Dual-Modality Coverage | **FULLY RESOLVED** | 975 / 1,007 subjects have both raw T1 and rs-fMRI on S3. `has_func_available` column added. |
| **9** | Site / Scanner Metadata | **FULLY RESOLVED** | 16-site reference table compiled at [`data/phenotypic/site_scanner_reference.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/site_scanner_reference.csv) (TR values from 0.475s to 3.000s). |
| **10**| Embedding Output Spec | **FULLY RESOLVED** | 3D ResNet-18 outputs a 512-dimensional structural feature vector (`(B, 512)`). |
