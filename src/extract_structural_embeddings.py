#!/usr/bin/env python3
"""
src/extract_structural_embeddings.py

Extracts 512-D Structural MRI Embeddings from Person 1's Trained 3D ResNet-18 Model Checkpoint
(models/checkpoints/best_resnet3d.pth) for all available preprocessed 3D T1 volumes.

Outputs:
- results/structural_embeddings.csv
- results/structural_embedding_manifest.csv
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Enforce UTF-8 output encoding for Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models.resnet3d import resnet18_3d

STRUCT_PREPROC_DIR = PROJECT_ROOT / "data" / "preprocessed" / "structural"
CKPT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_resnet3d.pth"
OUTPUT_DIR = PROJECT_ROOT / "results"
STRUCT_CSV_PATH = OUTPUT_DIR / "structural_embeddings.csv"
STRUCT_MANIFEST_PATH = OUTPUT_DIR / "structural_embedding_manifest.csv"
LOG_FILE = PROJECT_ROOT / "logs" / "structural_embedding_extraction.log"


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def extract_resnet3d_512d(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extracts 512-D feature vector from 3D ResNet-18 after global average pooling layer.
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)  # [B, 512]
    return x


def main():
    setup_logging()
    logging.info("=" * 80)
    logging.info("STRUCTURAL MRI 512-D EMBEDDING EXTRACTION INITIALIZED")
    logging.info("=" * 80)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"3D ResNet checkpoint missing: {CKPT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18_3d(num_classes=2, in_channels=1).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    model.eval()

    if not STRUCT_PREPROC_DIR.exists():
        raise FileNotFoundError(f"Structural preprocessed directory missing: {STRUCT_PREPROC_DIR}")

    files = sorted([f for f in os.listdir(STRUCT_PREPROC_DIR) if f.endswith(".npy")])
    logging.info(f"Found {len(files)} structural preprocessed .npy files in {STRUCT_PREPROC_DIR}")

    records = []
    manifest_records = []

    for f in files:
        sub_key = f.replace(".npy", "")
        parts = sub_key.split("_")
        site_code = f"{parts[0]}_{parts[1]}" if len(parts) >= 3 else parts[0]
        sub_id = parts[-1]

        npy_path = STRUCT_PREPROC_DIR / f
        arr = np.load(npy_path)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            emb_512 = extract_resnet3d_512d(model, tensor).cpu().numpy().flatten()

        rec = {
            "participant_id": sub_id,
            "site_code": site_code,
            "subject_key": sub_key
        }
        for d in range(512):
            rec[f"embedding_{d}"] = float(emb_512[d])
        records.append(rec)

        manifest_records.append({
            "participant_id": sub_id,
            "site_code": site_code,
            "subject_key": sub_key,
            "embedding_dimension": 512,
            "source_model": "3D ResNet-18 (best_resnet3d.pth)"
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_struct = pd.DataFrame(records)
    df_struct.to_csv(STRUCT_CSV_PATH, index=False)
    logging.info(f"Saved structural embeddings to: {STRUCT_CSV_PATH}")

    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(STRUCT_MANIFEST_PATH, index=False)
    logging.info(f"Saved structural manifest to: {STRUCT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
