# Project Rules and Constraints (`rules.md`)

**Project Title:** 3-Level Autism Severity Detection Using Multimodal MRI  
**Repository Location:** `d:\Projects\MRI_Autism\mri 2`  
**Applicability:** All AI Coding Assistants, Developers, and Maintainers  
**Document Version:** 1.0  

---

## 1. Executive Summary

This document establishes strict, non-negotiable operational rules and constraints for all human developers and AI coding agents working on this codebase. The purpose of these rules is to maintain scientific integrity, preserve verified model checkpoints, prevent data leakage, enforce clinical safety disclaimers, and eliminate data fabrication.

---

## 2. Mandatory Rules for AI Coding Agents

> [!IMPORTANT]
> All automated agents and developers **MUST** strictly adhere to the following 20 core directives:

1. **Inspect Existing Code Before Modifying**: Never write new functions or edit configuration files without inspecting authoritative source code and data schemas first.
2. **Preserve Completed Project Functionality**: Do not break, disable, or delete existing, working pipeline stages (such as T1 preprocessing, fMRI graph construction, GAT extraction, or multimodal fusion).
3. **Never Overwrite or Delete Valid Artifacts**: Existing model checkpoints (`.pth`, `.pt`), embedding CSVs, and JSON metric reports must never be deleted or overwritten without explicit user approval.
4. **Zero Data Fabrication Policy**: NEVER invent or fabricate synthetic MRI images, participant IDs, severity labels, model predictions, confidence probabilities, attention weights, connectivity values, or evaluation metrics.
5. **Use Actual Artifacts as Source of Truth**: All scientific numbers, metrics, and patient results rendered in reports or UI components must be read directly from verified project files in `results/`, `models/`, or `data/`.
6. **Strict Subject Matching by `participant_id`**: Match all subject-level data across structural, functional, prediction, and image sources using explicit string equality on `participant_id`. Never rely on dataframe index, row position, or array ordering.
7. **Preserve Master Data Splits**: The frozen train (102), validation (25), and held-out test (30) splits in `data/phenotypic/split_master_manifest.csv` must never be altered or re-randomized.
8. **No Training/Evaluation Leakage**: Never train, tune, or optimize hyperparameters on the held-out test set during model development.
9. **No Retraining for Frontend Edits**: Never trigger online model retraining, fine-tuning, or heavy batch inference merely to update UI components or layout styles in `app.py`.
10. **Preserve Model Architectures & Labels**: Do not alter the 3D ResNet-18, 62-node GAT, or Multimodal Attention Fusion architectures, loss functions, embedding dimensions (512-D), or severity labels unless explicitly instructed.
11. **Enforce Research Severity Terminology**: Always label target classes as **Low**, **Moderate**, and **High** ("Research-Defined Severity Classes"). Do NOT use "Clinical Level 1/2/3" or imply clinical DSM-5 diagnosis.
12. **Distinguish FC from GAT Attention**: Clearly distinguish raw Functional Connectivity (Fisher-z correlation matrices) from learned Graph Attention Network (GAT) attention coefficients in UI titles and text captions.
13. **Truthful Unavailable States**: If a participant-level graph, NIfTI slice, or prediction is unavailable, display a clear, truthful "Unavailable" notice instead of generating mock placeholders or silently substituting another subject's data.
14. **Metric Agreement**: Ensure that all displayed performance metrics match verified evaluation artifacts (Test Accuracy: 40.00%, Macro-F1: 0.3592, GAT Macro-F1: 0.3738, GCN Macro-F1: 0.2689).
15. **Mandatory Non-Clinical Disclaimer**: Maintain the visible academic research disclaimer on all frontend views and documentation outputs ("Academic Research Prototype — Not for Clinical Diagnosis").
16. **Modular & Maintainable Code**: Write clean, modular, and maintainable Python code adhering to PEP 8 standards with descriptive docstrings.
17. **Mandatory Pre-Commit Verification**: Never claim a task or bug fix is complete without running compilation (`python -m py_compile app.py`) or verification scripts.
18. **Dependency Control**: Do not introduce new third-party Python packages unless necessary, justified, and confirmed to be compatible with the environment.
19. **Privacy & Anonymization**: Preserve subject privacy by displaying only anonymized ABIDE II participant IDs (`28820`, `28838`, `29733`, etc.) without personal identifiable information.
20. **No Simplified Mock Code**: Never replace complex, working algorithmic implementations with simplified dummy functions or stubbed code.

---

## 3. Agent Operational Workflow Guidelines

Before making any modifications to this repository, an AI agent MUST follow this 3-phase execution protocol:

```
[ Phase 1: Pre-Modification Audit ]
  ├── 1. Read relevant project files using view_file or inspection scripts.
  ├── 2. Verify current git status and file integrity.
  └── 3. Identify exact source of truth artifacts for the targeted task.

[ Phase 2: Controlled Modification ]
  ├── 1. Apply targeted, minimal edits preserving existing contracts.
  ├── 2. Enforce strict participant_id matching for any data access.
  └── 3. Maintain dark navy theme styling and scientific disclaimers in UI.

[ Phase 3: Post-Modification Verification ]
  ├── 1. Run compilation check (python -m py_compile app.py).
  ├── 2. Execute verification script or data audit test.
  └── 3. Confirm zero metrics regressions and report exact results.
```
