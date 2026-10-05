# STEP 12K — RESEARCH LIMITATIONS & SCIENTIFIC DISCLAIMER

> [!CAUTION]
> **ACADEMIC RESEARCH PROTOTYPE DISCLAIMER**:
> This codebase and model represent an **academic research prototype** for 3-class autism severity classification. This model is **NOT** a clinical diagnostic tool, is **NOT** FDA-cleared, and MUST NOT be used for clinical diagnosis or medical decision-making.

---

## Key Research Limitations

1. **Small Sample Size**: The primary rs-fMRI and T1 structural dataset contains 157 subjects (102 train, 25 val, 30 test).
2. **Limited Test Set Support**: Final evaluation is based on 30 held-out test subjects (Low: 5, Moderate: 13, High: 12).
3. **Class Imbalance**: Low severity class has lower representation (19 train, 5 test), contributing to lower F1-score (0.1538).
4. **Site Heterogeneity**: Data spans 9 ABIDE-II scanner sites with varying acquisition protocols.
5. **Attention Interpretability**: Attention weights reflect learned neural network feature-gating signals, NOT verified causal biological mechanisms.
6. **External Validation Required**: Independent validation on outside datasets (e.g. ABIDE-I or new cohorts) is required before assessing generalizability.
