# Product Requirements Document (PRD)

**Project Title:** 3-Level Autism Severity Detection Using Multimodal MRI  
**Repository Location:** `d:\Projects\MRI_Autism\mri 2`  
**Status:** Completed Academic Research Prototype  
**Document Version:** 1.0  

---

## 1. Executive Overview

The **3-Level Autism Severity Detection Using Multimodal MRI** project is an academic research system designed to evaluate the integration of structural T1-weighted Magnetic Resonance Imaging (sMRI) and resting-state functional MRI (rs-fMRI) for classifying Autism Spectrum Disorder (ASD) into three research-defined severity classes: **Low**, **Moderate**, and **High**.

The platform combines deep 3D spatial feature extraction (3D ResNet-18) with graph-based functional connectivity modeling (Graph Attention Networks - GAT) and a Softmax Modality Attention Gating mechanism. An interactive Streamlit web application (`app.py`) serves as the scientific demonstration dashboard for exploring preprocessed brain volumes, functional graphs, model predictions, modality contribution splits, and explainability heatmaps.

---

## 2. Problem Statement & Research Motivation

### 2.1 Problem Statement
Autism Spectrum Disorder exhibits substantial clinical and neurobiological heterogeneity. Traditional machine learning models often rely on single-modality MRI (either structural anatomy or functional connectivity) or attempt binary classification (ASD vs Control). Single-modality approaches miss complementary information between structural brain alterations and functional circuit dysregulation, while binary classification fails to capture varying levels of symptom severity.

### 2.2 Research Objectives
1. **Multi-Level Severity Classification**: Classify subjects into three research-defined severity categories derived from ADOS-2 Calibrated Severity Scores (CSS).
2. **Multimodal MRI Fusion**: Combine 512-dimensional structural embeddings (3D ResNet-18) and 512-dimensional functional embeddings (62-node GAT) using an attention-gated fusion architecture.
3. **Modality Interpretability**: Quantify the relative reliance on structural vs functional neuroimaging features per participant and per severity class.
4. **Reproducible Benchmarking**: Benchmark the multimodal attention fusion model against single-modality baselines (Graph Convolutional Networks - GCN, and GAT).

---

## 3. Target Users & Primary Use Cases

### 3.1 Target User Groups
- **Academic Researchers**: Computational neuroscientists and biomedical imaging researchers investigating multimodal MRI integration and GNN architectures.
- **Students & Educators**: Learners studying deep learning in neuroimaging, graph neural networks, and explainable AI in medicine.
- **Research Teams**: Data science teams seeking reproducible benchmarks and dataset contracts on the ABIDE II dataset.

### 3.2 Primary Use Cases
1. **Interactive Cohort Exploration**: Inspect demographic distributions across ABIDE II acquisition sites, age, sex, and ADOS-2 severity scores.
2. **Single-Subject Neuroimaging Inspection**: Explore high-resolution 3D T1 MRI slices (Axial, Coronal, Sagittal) and 62-node Harvard-Oxford functional connectivity heatmaps for individual subjects.
3. **Modality Contribution Analysis**: Compare modality attention weights (\(\alpha_{\text{struct}}\) vs \(\alpha_{\text{func}}\)) to identify whether structural or functional features drive specific severity predictions.
4. **Model Benchmark Evaluation**: Compare multi-class classification performance metrics (Accuracy, Macro-F1, Weighted-F1, Precision, Recall, Confusion Matrices) across GCN, GAT, and Multimodal Fusion models.

---

## 4. Cohort & Dataset Definitions

To prevent sample leakage and maintain scientific integrity, the project strictly defines four cohort tiers:

```
+-------------------------------------------------------------------+
| FULL ABIDE II COHORT (N = 1,007)                                  |
| 516 ASD | 491 Neurotypical Controls                               |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
| ADOS-2 SEVERITY COHORT (N = 228)                                  |
| Subjects with valid ADOS-2 Calibrated Severity Scores (CSS)       |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
| FINAL MATCHED MULTIMODAL MODELING COHORT (N = 157)                |
| Matched Structural T1 + Functional fMRI (100% matched pair IDs)   |
| Low = 27 | Moderate = 66 | High = 64                              |
+-------------------------------------------------------------------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
         v                       v                       v
+------------------+   +-------------------+   +--------------------+
| TRAIN (N = 102)  |   | VAL (N = 25)      |   | HELD-OUT TEST (30) |
+------------------+   +-------------------+   +--------------------+
```

### Cohort Tier Summary
1. **Full ABIDE II Cohort (N = 1,007)**: Total phenotypic database across 19 acquisition sites (516 ASD, 491 Controls).
2. **ADOS-2 Severity Cohort (N = 228)**: Subset of ASD participants with complete ADOS-2 Calibrated Severity Scores.
3. **Final Matched Multimodal Cohort (N = 157)**: Subjects with verified, co-registered 3D T1 MRI volumes and 62-ROI rs-fMRI time-series.
   - **Low Severity Class (ADOS CSS 1–4)**: 27 subjects
   - **Moderate Severity Class (ADOS CSS 5–7)**: 66 subjects
   - **High Severity Class (ADOS CSS 8–10)**: 64 subjects
4. **Official Train / Validation / Test Splits**:
   - **Training Set**: 102 subjects
   - **Validation Set**: 25 subjects
   - **Held-Out Test Set**: 30 subjects (5 Low, 13 Moderate, 12 High)

---

## 5. Functional & Non-Functional Requirements

### 5.1 Functional Requirements
- **FR-1: Structural Processing**: Load T1 NIfTI volumes, apply N4 bias correction, skull stripping, MNI152 registration, 128×128×128 resampling, and extract 512-D spatial embeddings via 3D ResNet-18.
- **FR-2: Functional Processing**: Extract 62 Harvard-Oxford ROI time-series, compute Pearson functional connectivity with Fisher-z transformation, construct k-NN brain graphs, and extract 512-D embeddings via 2-layer GAT.
- **FR-3: Multimodal Attention Fusion**: Project 512-D structural and 512-D functional embeddings into 256-D latent spaces, compute Softmax modality attention weights, concatenate into a 1024-D representation, and output 3-class logits.
- **FR-4: Interactive Dashboard**: Provide an 8-workspace Streamlit frontend (`app.py`) for exploring predictions, MRI slices, FC heatmaps, brain graphs, and metrics.
- **FR-5: Dynamic Participant State**: Allow selection of any participant ID from the 157-subject cohort, updating all UI components without hardcoded IDs or silent fallbacks.

### 5.2 Non-Functional Requirements
- **NFR-1: Scientific Reproducibility**: Fixed random seeds (seed 42) across train/val/test splits and PyTorch models.
- **NFR-2: No On-The-Fly Model Inference**: Dashboard operates on pre-computed artifacts to ensure instant response times without background training.
- **NFR-3: High-Resolution Visualization**: Render authentic 3D NIfTI volumes using 1st–99th percentile intensity windowing and bicubic interpolation without pixelation.
- **NFR-4: Memory Efficiency**: Caching of heavy NIfTI load operations via Streamlit `@st.cache_data`.

---

## 6. System Inputs, Processing Pipeline, & Outputs

```
[ Inputs ]
  ├── T1 Structural MRI (NIfTI .nii.gz)
  ├── rs-fMRI BOLD Signal (NIfTI .nii.gz)
  └── Phenotypic ADOS-2 Scores & Demographic CSVs
        │
        v
[ Processing Pipeline ]
  ├── Structural: N4 Correction ➔ Skull Strip ➔ MNI 128x128x128 ➔ 3D ResNet-18 (512-D)
  ├── Functional: Preproc ➔ 62 HO ROIs ➔ Pearson FC ➔ Fisher-z ➔ GAT (512-D)
  └── Multimodal: Linear(512->256) ➔ Softmax Attention ➔ 1024-D Vector ➔ Classifier Head
        │
        v
[ Outputs ]
  ├── 3-Class Research Severity Prediction (Low / Moderate / High)
  ├── Modality Attention Split (Structural % vs Functional %)
  ├── Grad-CAM 3D Regional Activations Summary
  └── GAT Top Important Brain Regions & Connectivity Edges
```

---

## 7. Success Criteria & Verified Performance Benchmarks

All performance evaluations are based strictly on verified test set evaluation artifacts (`results/multimodal/final_metrics.json`, `models/fmri_gat/test_metrics.json`, `models/fmri_gcn/test_metrics.json`):

| Model Architecture | Input Modality | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GCN Baseline** | fMRI 62-Node Graph | 26.67% | 0.2864 | 0.2936 | 0.2689 | 0.2704 |
| **GAT Functional Model** | fMRI 62-Node Graph | 40.00% | 0.4273 | 0.3962 | 0.3738 | 0.4075 |
| **Multimodal Attention Fusion** | Structural + Functional | **40.00%** | **0.3693** | **0.3637** | **0.3592** | **0.4072** |

### Class-Wise Multimodal Performance (Held-Out Test Set N = 30)
- **Low Class (Support = 5)**: Precision = 0.1250, Recall = 0.2000, F1 = 0.1538
- **Moderate Class (Support = 13)**: Precision = 0.4444, Recall = 0.3077, F1 = 0.3636
- **High Class (Support = 12)**: Precision = 0.5385, Recall = 0.5833, F1 = 0.5600
- **Modality Reliance Split**: Structural = 48.92%, Functional = 51.08%

---

## 8. Limitations, Risks, & Future Roadmap

### 8.1 Limitations & Risks
- **Sample Size Constraints**: The matched multimodal cohort is restricted to N = 157 subjects due to T1/fMRI availability across ABIDE II sites.
- **Scanner Variance**: ABIDE II pools data across multiple institutions with varying scanner strengths (1.5T vs 3.0T) and TR settings.
- **Class Imbalance**: Low severity class has lower support (N=27 overall, N=5 in test set), leading to lower per-class F1 for Low severity.

### 8.2 Future Roadmap
- Expansion to larger datasets (ABIDE I, UK Biobank, SPARK).
- Integration of Diffusion Tensor Imaging (DTI) for white-matter tractography.
- Transition to dynamic functional connectivity (sliding-window time-varying graphs).

---

## 9. Explicit Non-Clinical Disclaimer

> [!CAUTION]
> **ACADEMIC RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL DIAGNOSIS**  
> This software system, model weights, predictions, and visualizations are strictly intended for academic research, education, and methodological demonstration. The system is **NOT** a clinical diagnostic tool and must **NOT** be used for patient screening, medical diagnosis, treatment planning, or clinical decision-making. The severity classes (**Low**, **Moderate**, **High**) represent research-defined score bins and are **NOT** equivalent to DSM-5 clinical diagnostic levels.
