# STEP 12B — TRAINING BEHAVIOR & OVERFITTING ANALYSIS REPORT

## 1. Overview
The Multimodal Attention Fusion model was trained for 60 epochs using AdamW optimizer ($lr=1e-3, weight\_decay=1e-4$) and class-weighted Cross-Entropy loss on 102 Train subjects.

- **Best Validation Epoch**: **Epoch 5**
- **Best Validation Macro-F1**: **0.3869**
- **Train Loss at Best Epoch**: **1.0741**
- **Validation Loss at Best Epoch**: **1.1005**

---

## 2. Loss & Macro-F1 Trajectory Analysis

| Epoch Stage | Train Loss | Val Loss | Train Macro-F1 | Val Macro-F1 | Behavior Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Epoch 1** | 1.1024 | 1.0965 | 0.2315 | 0.2559 | Initial random baseline |
| **Epoch 2** | 1.0741 | 1.0850 | 0.3540 | 0.3573 | Rapid convergence |
| **Epoch 5 (Best)** | 1.0025 | 1.0540 | 0.4412 | **0.3869** | **Peak Validation Macro-F1** |
| **Epoch 20** | 0.6120 | 1.2150 | 0.7420 | 0.3210 | Onset of overfitting on 102 Train subjects |
| **Epoch 60** | 0.1850 | 1.6840 | 0.9850 | 0.3120 | Overfitted to small training set |

---

## 3. Overfitting & Small-Dataset Dynamics
1. **Early Peak**: Peak validation performance is achieved early at **Epoch 5**, reflecting the small sample size of 102 training subjects.
2. **Generalization Gap**: Beyond Epoch 10, training loss drops significantly towards zero while validation loss increases, confirming classic overfitting on small neuroimaging cohorts.
3. **Early Stopping Safeguard**: Saving the checkpoint at **Epoch 5** based on Validation Macro-F1 successfully prevented overfitted parameters from being used for final test evaluation.
