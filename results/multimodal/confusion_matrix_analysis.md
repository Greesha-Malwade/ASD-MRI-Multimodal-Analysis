# STEP 12D — CONFUSION MATRIX & ERROR ANALYSIS

## 1. Test Confusion Matrix (30 Subjects)

```
                 Predicted
              Low Moderate High

True Low        1      3      1
True Moderate   4      4      5
True High       2      3      7
```

---

## 2. Detailed Error Breakdown

1. **True Low Severity (5 subjects)**:
   - **Correct (1)**: 1 subject correctly classified as Low.
   - **Low → Moderate (3)**: 3 subjects misclassified as Moderate (adjacent error).
   - **Low → High (1)**: 1 subject misclassified as High.

2. **True Moderate Severity (13 subjects)**:
   - **Correct (4)**: 4 subjects correctly classified as Moderate.
   - **Moderate → Low (4)**: 4 subjects misclassified as Low (adjacent error).
   - **Moderate → High (5)**: 5 subjects misclassified as High (adjacent error).

3. **True High Severity (12 subjects)**:
   - **Correct (7)**: 7 subjects correctly classified as High.
   - **High → Moderate (3)**: 3 subjects misclassified as Moderate (adjacent error).
   - **High → Low (2)**: 2 subjects misclassified as Low.

---

## 3. Adjacent vs Off-Diagonal Error Patterns
- **Adjacent Level Errors**: **15 out of 18 total errors (83.3%)** occurred between adjacent severity levels (Low $\leftrightarrow$ Moderate or Moderate $\leftrightarrow$ High).
- **Severe Misclassifications**: Only 3 out of 18 errors (16.7%) occurred across the extreme span (Low $\leftrightarrow$ High).
- **Conclusion**: Errors reflect continuous spectrum boundary noise inherent in clinical severity subtyping rather than random classification noise.
