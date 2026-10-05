#!/usr/bin/env python3
"""
src/step12_final_evaluation.py

Stage 12A — Final Evaluation Audit & Re-computation Script for Multimodal Attention Fusion.

Re-computes metrics independently from results/multimodal/test_predictions.csv
and verifies consistency against results/multimodal/final_metrics.json.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

PRED_CSV_PATH = PROJECT_ROOT / "results" / "multimodal" / "test_predictions.csv"
METRICS_JSON_PATH = PROJECT_ROOT / "results" / "multimodal" / "final_metrics.json"
HISTORY_CSV_PATH = PROJECT_ROOT / "results" / "multimodal" / "training_history.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "multimodal" / "best_multimodal_attention.pt"

SEVERITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}
REVERSE_MAP = {0: "Low", 1: "Moderate", 2: "High"}


def main():
    print("=" * 80)
    print("STEP 12A: FINAL MODEL EVALUATION AUDIT")
    print("=" * 80)

    # 1. Check Artifact Availability
    assert MODEL_PATH.exists(), f"Missing model checkpoint: {MODEL_PATH}"
    assert PRED_CSV_PATH.exists(), f"Missing test predictions: {PRED_CSV_PATH}"
    assert METRICS_JSON_PATH.exists(), f"Missing final metrics JSON: {METRICS_JSON_PATH}"
    assert HISTORY_CSV_PATH.exists(), f"Missing training history CSV: {HISTORY_CSV_PATH}"

    print(f"[PASS] All required input artifacts exist.")

    # 2. Read Test Predictions
    df_preds = pd.read_csv(PRED_CSV_PATH)
    print(f"Total test prediction rows: {len(df_preds)}")
    assert len(df_preds) == 30, f"Expected 30 test subjects, got {len(df_preds)}"
    assert df_preds["participant_id"].nunique() == 30, "Duplicate participant IDs found in test set!"

    y_true = np.array([SEVERITY_MAP[c] for c in df_preds["true_class"]])
    y_pred = np.array([SEVERITY_MAP[c] for c in df_preds["predicted_class"]])

    # 3. Independently Recompute Metrics
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\nIndependently Re-computed Test Metrics:")
    print(f"  Test Accuracy:     {acc:.4f} ({int(acc*30)}/30 correct)")
    print(f"  Macro Precision:   {p_macro:.4f}")
    print(f"  Macro Recall:      {r_macro:.4f}")
    print(f"  Macro F1:          {f1_macro:.4f}")
    print(f"  Weighted F1:       {f1_w:.4f}")

    print("\nClass-wise Re-computed Metrics:")
    for c_idx, c_name in REVERSE_MAP.items():
        print(f"  {c_name:10s} (Support={sup_per[c_idx]:2d}): Precision={p_per[c_idx]:.4f}, Recall={r_per[c_idx]:.4f}, F1={f1_per[c_idx]:.4f}")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=["True Low", "True Moderate", "True High"], columns=["Pred Low", "Pred Moderate", "Pred High"]))

    # 4. Compare with Saved final_metrics.json
    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    print("\nComparing Re-computed vs Saved final_metrics.json:")
    diff_acc = abs(acc - saved_metrics["test_accuracy"])
    diff_f1 = abs(f1_macro - saved_metrics["test_macro_f1"])

    print(f"  Accuracy Difference: {diff_acc:.8f}")
    print(f"  Macro-F1 Difference: {diff_f1:.8f}")

    if diff_acc < 1e-6 and diff_f1 < 1e-6:
        print("\n[VERIFICATION PASS] Independently re-computed metrics MATCH saved final_metrics.json 100% perfectly!")
    else:
        print("\n[WARNING] Discrepancy detected between re-computed and saved metrics!")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
