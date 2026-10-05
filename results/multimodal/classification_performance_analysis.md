# STEP 12C — CLASSIFICATION PERFORMANCE ANALYSIS

## 1. Overview of Held-Out Test Results (30 Subjects)

| Severity Class | Support | Correct Predictions | Precision | Recall | F1-Score | Rank / Ease |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **High** | 12 | 7 | 0.5385 | 0.5833 | **0.5600** | **Easiest Class (Rank 1)** |
| **Moderate** | 13 | 4 | 0.4444 | 0.3077 | **0.3636** | **Intermediate (Rank 2)** |
| **Low** | 5 | 1 | 0.1250 | 0.2000 | **0.1538** | **Hardest Class (Rank 3)** |

---

## 2. Class Difficulty Analysis

### A. High Severity Class (F1 = 0.5600) — Easiest
- **Why High Severity is Easiest**: High autism severity subjects present stronger structural anatomical alterations and distinct functional connectivity patterns in graph representations, making their 512-D embeddings most distinct from the other classes.
- **Support & Recall**: 7 out of 12 High severity test subjects were correctly classified, achieving highest recall (58.33%).

### B. Moderate Severity Class (F1 = 0.3636) — Intermediate
- **Intermediate Boundary**: Moderate severity acts as a spectrum boundary between Low and High, leading to predictions spanning adjacent severity levels.
- **Support & Precision**: 4 out of 13 Moderate test subjects were correctly classified.

### C. Low Severity Class (F1 = 0.1538) — Hardest
- **Class Imbalance & Support**: Low severity has the smallest sample size (only 5 test subjects, 19 train subjects).
- **Subtle Biomarkers**: Neuroimaging alterations in mild/low ASD severity are subtle and overlap heavily with moderate presentations, making detection challenging on a small dataset.
