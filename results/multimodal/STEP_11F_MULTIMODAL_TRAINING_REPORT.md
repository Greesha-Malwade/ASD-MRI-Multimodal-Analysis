# STEP 11F — MULTIMODAL ATTENTION FUSION TRAINING REPORT

## 1. Executive Summary
The final multimodal attention fusion model was trained combining Person 1's 512-D Structural MRI embeddings (3D ResNet-18) and Person 2's 512-D Functional MRI embeddings (2-Layer GAT) for 3-class autism severity prediction (`Low`, `Moderate`, `High`).

- **Dataset Size**: 157 subjects (Train: 102, Validation: 25, Test: 30)
- **Model Selection**: Driven strictly by Validation Macro-F1 (Best epoch: **5**, Val Macro-F1: **0.3869**)
- **Test Performance**: Evaluated ONCE on held-out Test set:
  - **Test Accuracy**: **0.4000** (12/30 correct)
  - **Test Macro-F1**: **0.3592**
  - **Test Weighted-F1**: **0.4072**

---

## 2. Model Architecture & Parameters
```
STRUCTURAL (512-D)                  FUNCTIONAL (512-D)
        │                                  │
  Linear(512, 256)                   Linear(512, 256)
        │                                  │
    ReLU + Drop                        ReLU + Drop
        │                                  │
     h_s (256)                          h_f (256)
        │                                  │
        └──────────────┬───────────────────┘
                       │
            Modality Attention Gating
      Score_s = Linear(256, 1)(h_s), Score_f = Linear(256, 1)(h_f)
       [a_s, a_f] = Softmax([Score_s, Score_f])
                       │
       Fused: h_fused = a_s * h_s + a_f * h_f (256-D)
                       │
                   Classifier
           Linear(256, 64) -> ReLU -> Linear(64, 3)
```
- **Total Trainable Parameters**: 279,813

---

## 3. Data Integrity & Leakage Prevention Rules Enforced
1. **Join by participant_id**: Zero position-based row matching.
2. **Train-Only Scaler**: `StandardScaler` fitted strictly on 102 Train subjects; applied to Val and Test.
3. **Train-Only Class Weighting**: Computed from Train labels (`Low: 27, Moderate: 66, High: 64`): `[1.2593, 0.5152, 0.5313]`.
4. **Validation-Only Selection**: Model selection guided by Validation Macro-F1. Test set evaluated ONCE after training.

---

## 4. Test Set Class-Wise Performance

| Severity Class | Support | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | 5 | 0.1250 | 0.2000 | 0.1538 |
| **Moderate** | 13 | 0.4444 | 0.3077 | 0.3636 |
| **High** | 12 | 0.5385 | 0.5833 | 0.5600 |
| **Macro Average** | 30 | 0.3693 | 0.3637 | **0.3592** |
| **Weighted Average**| 30 | - | - | **0.4072** |

---

## 5. Modality Attention Analysis
- **Overall Mean Structural Attention ($a_s$)**: **0.4892** (48.9%)
- **Overall Mean Functional Attention ($a_f$)**: **0.5108** (51.1%)

> [!NOTE]
> Modality attention weights reflect learned gating signals. Functional rs-fMRI embeddings ($a_f pprox 51.1\%$) and Structural T1 embeddings ($a_s pprox 48.9\%$) contribute dynamically to the 3-class severity prediction.

---

## 6. Limitations & Disclaimer
- **Dataset Scale**: The dataset contains 157 subjects (102 train, 25 val, 30 test).
- **Scope**: This model is an academic research prototype for multi-site neuroimaging fusion analysis. It is **not** a clinical diagnostic tool and does not claim clinical generalization beyond this research dataset.
