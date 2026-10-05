#!/usr/bin/env python3
"""
src/analyze_biomarkers.py
Statistical Biomarker Analysis & Feature Significance Testing for ASD vs. TC Structural Features.

- Computes Mann-Whitney U tests and Cohen's d effect sizes for all 93 structural features
- Applies Benjamini-Hochberg False Discovery Rate (FDR q < 0.05) correction
- Exports top biomarker rankings to results/top_structural_biomarkers.csv
- Plots top-15 feature importance chart to results/feature_importance.png
"""

import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def cohen_d(x1: np.ndarray, x2: np.ndarray) -> float:
    n1, n2 = len(x1), len(x2)
    if n1 <= 1 or n2 <= 1:
        return 0.0
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) + 1e-8
    return float((np.mean(x1) - np.mean(x2)) / s_pooled)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    q_values = np.zeros(n)

    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q_val = (sorted_p[i] * n) / rank
        cum_min = min(cum_min, q_val)
        q_values[sorted_indices[i]] = min(1.0, cum_min)

    return q_values


def analyze_structural_biomarkers(df_feats: pd.DataFrame):
    meta_cols = [
        "subject_key", "site_code", "participant_id", "dx_group",
        "ados_2_severity_total", "age_at_scan", "sex", "raw_site_name", "site_id"
    ]
    feature_cols = [c for c in df_feats.columns if c not in meta_cols]

    # Mask ASD (dx_group == 1) vs TC (dx_group == 2)
    df_asd = df_feats[df_feats["dx_group"].astype(str) == "1"]
    df_tc = df_feats[df_feats["dx_group"].astype(str) == "2"]

    biomarker_results = []
    p_vals = []

    for col in feature_cols:
        v_asd = pd.to_numeric(df_asd[col], errors="coerce").dropna().values
        v_tc = pd.to_numeric(df_tc[col], errors="coerce").dropna().values

        if len(v_asd) > 0 and len(v_tc) > 0:
            u_stat, p_val = stats.mannwhitneyu(v_asd, v_tc, alternative="two-sided")
            d_val = cohen_d(v_asd, v_tc)
            mean_asd, std_asd = np.mean(v_asd), np.std(v_asd)
            mean_tc, std_tc = np.mean(v_tc), np.std(v_tc)
        else:
            u_stat, p_val, d_val = 0.0, 1.0, 0.0
            mean_asd, std_asd, mean_tc, std_tc = 0.0, 0.0, 0.0, 0.0

        p_vals.append(p_val)

        clean_name = col.replace("vol_ho_cort_", "").replace("vol_ho_sub_", "").replace("vol_", "").replace("_", " ").title()

        biomarker_results.append({
            "Feature Name": clean_name,
            "Raw Column": col,
            "ASD Mean ± SD": f"{mean_asd:.2f} ± {std_asd:.2f}",
            "Control Mean ± SD": f"{mean_tc:.2f} ± {std_tc:.2f}",
            "Cohen's d": round(d_val, 4),
            "Mann-Whitney U": round(u_stat, 2),
            "p-value": p_val,
        })

    # Apply BH-FDR correction
    q_vals = benjamini_hochberg(np.array(p_vals))

    for i, res in enumerate(biomarker_results):
        res["FDR q-value"] = q_vals[i]
        res["Significant (q < 0.05)"] = "Yes" if q_vals[i] < 0.05 else "No"
        res["p-value"] = f"{res['p-value']:.4e}"
        res["FDR q-value"] = f"{res['FDR q-value']:.4e}"

    df_bio = pd.DataFrame(biomarker_results)
    df_bio["abs_d"] = df_bio["Cohen's d"].abs()
    df_bio = df_bio.sort_values(by="abs_d", ascending=False).drop(columns=["abs_d"])

    return df_bio


def plot_top_feature_importance(df_bio: pd.DataFrame, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    top_df = df_bio.head(15).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(top_df))
    effects = top_df["Cohen's d"].astype(float).values
    colors = ["#2b5c8f" if e >= 0 else "#d95f02" for e in effects]

    ax.barh(y_pos, effects, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["Feature Name"], fontsize=10)
    ax.invert_yaxis()  # top feature on top
    ax.set_xlabel("Cohen's d Effect Size (ASD vs. Control)", fontsize=12)
    ax.set_title("Top 15 Structural MRI Biomarkers for ASD", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved feature importance plot to {out_png}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    log_file = project_root / "logs" / "analyze_biomarkers.log"
    setup_logging(log_file)

    logging.info("Starting Structural MRI Biomarker Statistical Significance Analysis...")

    features_csv = project_root / "data" / "features" / "structural_features.csv"
    results_dir = project_root / "results"
    out_csv = results_dir / "top_structural_biomarkers.csv"
    out_png = results_dir / "feature_importance.png"

    if not features_csv.exists():
        logging.error(f"Feature matrix missing at {features_csv}. Run Stage 3 first!")
        sys.exit(1)

    df_feats = pd.read_csv(features_csv)
    df_bio = analyze_structural_biomarkers(df_feats)
    df_bio.to_csv(out_csv, index=False)
    logging.info(f"Saved biomarker analysis ranking table to {out_csv}")

    plot_top_feature_importance(df_bio, out_png)

    print("\n" + "=" * 80)
    print("TOP 15 STRUCTURAL MRI BIOMARKERS FOR ASD (BY EFFECT SIZE)")
    print("=" * 80)
    print(df_bio[["Feature Name", "ASD Mean ± SD", "Control Mean ± SD", "Cohen's d", "FDR q-value"]].head(15).to_string(index=False))
    print(f"\nFull biomarker table saved to: {out_csv}")
    print(f"Feature importance plot saved to: {out_png}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
