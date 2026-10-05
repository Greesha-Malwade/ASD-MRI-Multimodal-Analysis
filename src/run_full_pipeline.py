#!/usr/bin/env python3
"""
src/run_full_pipeline.py
Unified Command-Line Orchestrator for Structural MRI ABIDE II Pipeline.

Usage:
  python src/run_full_pipeline.py --stage ALL
  python src/run_full_pipeline.py --stage 1
  python src/run_full_pipeline.py --stage 2 --test_n 5
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Unified Structural MRI Pipeline Orchestrator")
    parser.add_argument(
        "--stage",
        type=str,
        default="ALL",
        choices=["1", "2", "3", "4", "5", "6", "ALL"],
        help="Stage to execute (1: Prep, 2: Preprocessing, 3: Features, 4: ML Baselines, 5: 3D ResNet, 6: Grad-CAM, ALL: Complete Pipeline)",
    )
    parser.add_argument("--test_n", type=int, default=None, help="Number of subjects for test run (Stage 2)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for 3D ResNet training (Stage 5)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    py_exec = sys.executable

    def run_script(script_name: str, extra_args: list = []):
        cmd = [py_exec, str(project_root / "src" / script_name)] + extra_args
        print(f"\n>>> Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=project_root)
        if res.returncode != 0:
            print(f"ERROR: Script {script_name} failed with return code {res.returncode}")
            sys.exit(res.returncode)

    if args.stage in ["1", "ALL"]:
        run_script("dataset_prep.py")

    if args.stage in ["2", "ALL"]:
        extra = ["--test_n", str(args.test_n)] if args.test_n else []
        run_script("preprocess_t1.py", extra)

    if args.stage in ["3", "ALL"]:
        run_script("extract_features.py")

    if args.stage in ["4", "ALL"]:
        run_script("baseline_models.py")

    if args.stage in ["5", "ALL"]:
        run_script("train_resnet3d.py", ["--epochs", str(args.epochs)])

    if args.stage in ["6", "ALL"]:
        run_script("gradcam_3d.py", ["--num_subjects", "5"])

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
