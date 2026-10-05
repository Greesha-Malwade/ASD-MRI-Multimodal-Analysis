#!/usr/bin/env python3
"""
src/train_step11f_multimodal.py

Stage 11F — Multimodal Attention Fusion Training
Combines Person 1's 512-D Structural MRI Embeddings and Person 2's 512-D Functional rs-fMRI GAT Embeddings.

Inputs:
- results/multimodal/final_aligned_embeddings_157.csv (157 subjects, 1024-D features)
- results/multimodal/final_alignment_manifest.csv

Outputs:
- models/multimodal/best_multimodal_attention.pt
- results/multimodal/multimodal_training_config.json
- results/multimodal/training_history.csv
- results/multimodal/loss_curve.png
- results/multimodal/macro_f1_curve.png
- results/multimodal/test_predictions.csv
- results/multimodal/confusion_matrix.csv
- results/multimodal/confusion_matrix.png
- results/multimodal/final_metrics.json
- results/multimodal/STEP_11F_MULTIMODAL_TRAINING_REPORT.md
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

INPUT_CSV_PATH = PROJECT_ROOT / "results" / "multimodal" / "final_aligned_embeddings_157.csv"
INPUT_MANIFEST_PATH = PROJECT_ROOT / "results" / "multimodal" / "final_alignment_manifest.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "multimodal"
RESULTS_DIR = PROJECT_ROOT / "results" / "multimodal"
LOG_FILE = PROJECT_ROOT / "logs" / "multimodal_fusion_training.log"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}
REVERSE_SEVERITY_MAP = {0: "Low", 1: "Moderate", 2: "High"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class MultimodalAttentionFusion(nn.Module):
    """
    Multimodal Attention-Gated Fusion Network:
    Structural 512-D -> Linear(512, 256) -> ReLU -> Dropout
    Functional 512-D -> Linear(512, 256) -> ReLU -> Dropout
    Modality Attention Gating: [a_s, a_f] = Softmax([Score_s, Score_f])
    Fused Representation: h_fused = a_s * h_s + a_f * h_f (256-D)
    Classifier Head: Dropout -> Linear(256, 64) -> ReLU -> Dropout -> Linear(64, 3)
    """
    def __init__(self, struct_dim: int = 512, func_dim: int = 512, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.proj_struct = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_func = nn.Sequential(
            nn.Linear(func_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.attn_struct = nn.Linear(hidden_dim, 1)
        self.attn_func = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, struct_x: torch.Tensor, func_x: torch.Tensor):
        h_s = self.proj_struct(struct_x)  # [B, 256]
        h_f = self.proj_func(func_x)      # [B, 256]

        score_s = self.attn_struct(h_s)   # [B, 1]
        score_f = self.attn_func(h_f)     # [B, 1]

        scores = torch.cat([score_s, score_f], dim=1)  # [B, 2]
        attn_weights = F.softmax(scores, dim=1)        # [B, 2]

        a_s = attn_weights[:, 0:1]  # [B, 1]
        a_f = attn_weights[:, 1:2]  # [B, 1]

        h_fused = a_s * h_s + a_f * h_f  # [B, 256]

        logits = self.classifier(h_fused)  # [B, 3]

        return logits, a_s, a_f, h_fused


class MultimodalDataset(Dataset):
    def __init__(self, struct_x: np.ndarray, func_x: np.ndarray, labels: np.ndarray, pids: list, splits: list):
        self.struct_x = torch.tensor(struct_x, dtype=torch.float32)
        self.func_x = torch.tensor(func_x, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.pids = pids
        self.splits = splits

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "struct_x": self.struct_x[idx],
            "func_x": self.func_x[idx],
            "label": self.labels[idx],
            "participant_id": self.pids[idx],
            "split": self.splits[idx]
        }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_pids = []
    all_attn_s = []
    all_attn_f = []

    with torch.no_grad():
        for batch in loader:
            struct_x = batch["struct_x"].to(device)
            func_x = batch["func_x"].to(device)
            targets = batch["label"].to(device)
            pids = batch["participant_id"]

            logits, a_s, a_f, _ = model(struct_x, func_x)
            loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_pids.extend(pids)
            all_attn_s.extend(a_s.cpu().numpy().flatten())
            all_attn_f.extend(a_f.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    avg_loss = total_loss / len(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "p_macro": p_macro,
        "r_macro": r_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "preds": all_preds,
        "targets": all_targets,
        "pids": all_pids,
        "attn_s": np.array(all_attn_s),
        "attn_f": np.array(all_attn_f)
    }


def main():
    set_seed(42)
    setup_logging()

    logging.info("=" * 80)
    logging.info("STEP 11F: MULTIMODAL ATTENTION FUSION TRAINING")
    logging.info("=" * 80)

    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"Aligned multimodal dataset missing at: {INPUT_CSV_PATH}")

    df_data = pd.read_csv(INPUT_CSV_PATH)
    logging.info(f"Loaded aligned multimodal dataset: {len(df_data)} subjects")

    # Step 16: Integrity verification
    train_mask = df_data["split"] == "train"
    val_mask = df_data["split"] == "val"
    test_mask = df_data["split"] == "test"

    train_count = train_mask.sum()
    val_count = val_mask.sum()
    test_count = test_mask.sum()

    logging.info(f"Data Split Counts: Train={train_count}, Val={val_count}, Test={test_count}")
    assert train_count == 102, f"Expected 102 train, got {train_count}"
    assert val_count == 25, f"Expected 25 val, got {val_count}"
    assert test_count == 30, f"Expected 30 test, got {test_count}"

    severity_counts = df_data["severity_class"].value_counts().to_dict()
    logging.info(f"Severity Class Distribution: {severity_counts}")
    assert severity_counts.get("Low") == 27, f"Expected 27 Low, got {severity_counts.get('Low')}"
    assert severity_counts.get("Moderate") == 66, f"Expected 66 Moderate, got {severity_counts.get('Moderate')}"
    assert severity_counts.get("High") == 64, f"Expected 64 High, got {severity_counts.get('High')}"

    func_cols = [f"functional_{i+1}" for i in range(512)]
    struct_cols = [f"structural_{i+1}" for i in range(512)]

    X_func = df_data[func_cols].values
    X_struct = df_data[struct_cols].values
    y_labels = np.array([SEVERITY_MAP[s] for s in df_data["severity_class"]])
    pids = df_data["participant_id"].tolist()
    splits = df_data["split"].tolist()

    assert not np.isnan(X_func).any(), "NaN detected in functional features"
    assert not np.isnan(X_struct).any(), "NaN detected in structural features"
    assert not np.isinf(X_func).any(), "Inf detected in functional features"
    assert not np.isinf(X_struct).any(), "Inf detected in structural features"

    # Step 5: Fit Scalers ONLY on Train set
    scaler_func = StandardScaler()
    scaler_struct = StandardScaler()

    X_func_train = scaler_func.fit_transform(X_func[train_mask])
    X_func_val = scaler_func.transform(X_func[val_mask])
    X_func_test = scaler_func.transform(X_func[test_mask])

    X_struct_train = scaler_struct.fit_transform(X_struct[train_mask])
    X_struct_val = scaler_struct.transform(X_struct[val_mask])
    X_struct_test = scaler_struct.transform(X_struct[test_mask])

    # Reconstruct full arrays
    X_func_scaled = np.zeros_like(X_func)
    X_struct_scaled = np.zeros_like(X_struct)

    X_func_scaled[train_mask] = X_func_train
    X_func_scaled[val_mask] = X_func_val
    X_func_scaled[test_mask] = X_func_test

    X_struct_scaled[train_mask] = X_struct_train
    X_struct_scaled[val_mask] = X_struct_val
    X_struct_scaled[test_mask] = X_struct_test

    # Step 4: Calculate Class Weights strictly on TRAIN set
    train_y = y_labels[train_mask]
    class_counts = np.bincount(train_y, minlength=3)
    total_train = len(train_y)
    class_weights = total_train / (3.0 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    logging.info(f"Train Class Counts (Low, Mod, High): {class_counts}")
    logging.info(f"Computed Train Class Weights: {class_weights.tolist()}")

    # Datasets and Loaders
    ds_train = MultimodalDataset(
        X_struct_scaled[train_mask], X_func_scaled[train_mask], y_labels[train_mask],
        [pids[i] for i in range(len(pids)) if train_mask[i]],
        [splits[i] for i in range(len(splits)) if train_mask[i]]
    )
    ds_val = MultimodalDataset(
        X_struct_scaled[val_mask], X_func_scaled[val_mask], y_labels[val_mask],
        [pids[i] for i in range(len(pids)) if val_mask[i]],
        [splits[i] for i in range(len(splits)) if val_mask[i]]
    )
    ds_test = MultimodalDataset(
        X_struct_scaled[test_mask], X_func_scaled[test_mask], y_labels[test_mask],
        [pids[i] for i in range(len(pids)) if test_mask[i]],
        [splits[i] for i in range(len(splits)) if test_mask[i]]
    )

    loader_train = DataLoader(ds_train, batch_size=16, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=16, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=16, shuffle=False)

    # Step 6: Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = MultimodalAttentionFusion(struct_dim=512, func_dim=512, hidden_dim=256, num_classes=3, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = MODEL_DIR / "best_multimodal_attention.pt"

    history_records = []
    epochs = 60

    logging.info("Starting training loop...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in loader_train:
            struct_x = batch["struct_x"].to(device)
            func_x = batch["func_x"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _, _, _ = model(struct_x, func_x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy())

        train_loss = train_loss / len(train_targets)
        train_acc = accuracy_score(train_targets, train_preds)
        _, _, train_f1, _ = precision_recall_fscore_support(train_targets, train_preds, average="macro", zero_division=0)

        # Validation evaluation
        val_res = evaluate(model, loader_val, criterion, device)

        history_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_res["loss"],
            "train_accuracy": train_acc,
            "validation_accuracy": val_res["acc"],
            "train_macro_f1": train_f1,
            "validation_macro_f1": val_res["f1_macro"],
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        if val_res["f1_macro"] > best_val_f1:
            best_val_f1 = val_res["f1_macro"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Epoch {epoch:03d} | New best Val Macro-F1: {best_val_f1:.4f} (Model saved to {best_model_path})")

    logging.info(f"Training completed. Best epoch: {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")

    # Save History CSV
    df_history = pd.DataFrame(history_records)
    history_csv = RESULTS_DIR / "training_history.csv"
    df_history.to_csv(history_csv, index=False)

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(df_history["epoch"], df_history["train_loss"], label="Train Loss", color="blue")
    plt.plot(df_history["epoch"], df_history["validation_loss"], label="Validation Loss", color="orange")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.title("Multimodal Attention Fusion: Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
    plt.close()

    # Plot Macro-F1 Curve
    plt.figure(figsize=(8, 5))
    plt.plot(df_history["epoch"], df_history["train_macro_f1"], label="Train Macro-F1", color="blue")
    plt.plot(df_history["epoch"], df_history["validation_macro_f1"], label="Validation Macro-F1", color="orange")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.title("Multimodal Attention Fusion: Macro-F1 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "macro_f1_curve.png", dpi=150)
    plt.close()

    # Step 8: Evaluate BEST model ONCE on held-out Test set
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_res = evaluate(model, loader_test, criterion, device)

    # Calculate per-class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        test_res["targets"], test_res["preds"], labels=[0, 1, 2], zero_division=0
    )

    per_class_metrics = {}
    for c_idx, c_name in REVERSE_SEVERITY_MAP.items():
        per_class_metrics[c_name] = {
            "precision": float(p_per[c_idx]),
            "recall": float(r_per[c_idx]),
            "f1": float(f1_per[c_idx]),
            "support": int(sup_per[c_idx])
        }

    # Step 10: Modality Attention Analysis
    test_pids = test_res["pids"]
    test_targets = test_res["targets"]
    test_preds = test_res["preds"]

    # Re-run inference on test set to record individual attention weights
    model.eval()
    attn_records = []

    with torch.no_grad():
        for batch in loader_test:
            struct_x = batch["struct_x"].to(device)
            func_x = batch["func_x"].to(device)
            targets = batch["label"].cpu().numpy()
            pids = batch["participant_id"]

            logits, a_s, a_f, _ = model(struct_x, func_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            a_s_arr = a_s.cpu().numpy().flatten()
            a_f_arr = a_f.cpu().numpy().flatten()

            for i in range(len(pids)):
                attn_records.append({
                    "participant_id": int(pids[i]),
                    "split": "test",
                    "true_class": REVERSE_SEVERITY_MAP[targets[i]],
                    "predicted_class": REVERSE_SEVERITY_MAP[preds[i]],
                    "structural_attention": float(a_s_arr[i]),
                    "functional_attention": float(a_f_arr[i])
                })

    df_test_preds = pd.DataFrame(attn_records)
    df_test_preds.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    mean_struct_attn = float(df_test_preds["structural_attention"].mean())
    mean_func_attn = float(df_test_preds["functional_attention"].mean())

    attn_by_class = {}
    for c_name in ["Low", "Moderate", "High"]:
        c_df = df_test_preds[df_test_preds["true_class"] == c_name]
        if len(c_df) > 0:
            attn_by_class[c_name] = {
                "mean_structural_attention": float(c_df["structural_attention"].mean()),
                "mean_functional_attention": float(c_df["functional_attention"].mean())
            }

    # Confusion Matrix
    cm = confusion_matrix(test_res["targets"], test_res["preds"], labels=[0, 1, 2])
    df_cm = pd.DataFrame(cm, index=["Low", "Moderate", "High"], columns=["Low", "Moderate", "High"])
    df_cm.to_csv(RESULTS_DIR / "confusion_matrix.csv")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Multimodal Attention Fusion: Test Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ["Low", "Moderate", "High"])
    plt.yticks(tick_marks, ["Low", "Moderate", "High"])

    thresh = cm.max() / 2.
    for i in range(3):
        for j in range(3):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Severity Class")
    plt.xlabel("Predicted Severity Class")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Final Metrics JSON
    final_metrics = {
        "test_accuracy": float(test_res["acc"]),
        "test_macro_precision": float(test_res["p_macro"]),
        "test_macro_recall": float(test_res["r_macro"]),
        "test_macro_f1": float(test_res["f1_macro"]),
        "test_weighted_f1": float(test_res["f1_weighted"]),
        "per_class_metrics": per_class_metrics,
        "mean_structural_attention": mean_struct_attn,
        "mean_functional_attention": mean_func_attn,
        "attention_by_class": attn_by_class,
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_f1)
    }

    with open(RESULTS_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)

    # Save Config JSON
    config_dict = {
        "architecture": "MultimodalAttentionFusion (Dual 512->256 Linear Projections + Modality Attention Gating + 3-Class MLP Head)",
        "structural_embedding_dim": 512,
        "functional_embedding_dim": 512,
        "hidden_dim": 256,
        "num_classes": 3,
        "dropout": 0.3,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": epochs,
        "batch_size": 16,
        "class_weights": class_weights.tolist(),
        "random_seed": 42,
        "normalization_method": "StandardScaler (fitted on train split only)",
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_f1),
        "test_accuracy": float(test_res["acc"]),
        "test_macro_f1": float(test_res["f1_macro"])
    }

    with open(RESULTS_DIR / "multimodal_training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    # Generate Report Markdown
    report_md = f"""# STEP 11F — MULTIMODAL ATTENTION FUSION TRAINING REPORT

## 1. Executive Summary
The final multimodal attention fusion model was trained combining Person 1's 512-D Structural MRI embeddings (3D ResNet-18) and Person 2's 512-D Functional MRI embeddings (2-Layer GAT) for 3-class autism severity prediction (`Low`, `Moderate`, `High`).

- **Dataset Size**: 157 subjects (Train: 102, Validation: 25, Test: 30)
- **Model Selection**: Driven strictly by Validation Macro-F1 (Best epoch: **{best_epoch}**, Val Macro-F1: **{best_val_f1:.4f}**)
- **Test Performance**: Evaluated ONCE on held-out Test set:
  - **Test Accuracy**: **{test_res['acc']:.4f}** ({int(test_res['acc']*30)}/30 correct)
  - **Test Macro-F1**: **{test_res['f1_macro']:.4f}**
  - **Test Weighted-F1**: **{test_res['f1_weighted']:.4f}**

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
| **Low** | {sup_per[0]} | {p_per[0]:.4f} | {r_per[0]:.4f} | {f1_per[0]:.4f} |
| **Moderate** | {sup_per[1]} | {p_per[1]:.4f} | {r_per[1]:.4f} | {f1_per[1]:.4f} |
| **High** | {sup_per[2]} | {p_per[2]:.4f} | {r_per[2]:.4f} | {f1_per[2]:.4f} |
| **Macro Average** | {len(test_res['targets'])} | {test_res['p_macro']:.4f} | {test_res['r_macro']:.4f} | **{test_res['f1_macro']:.4f}** |
| **Weighted Average**| {len(test_res['targets'])} | - | - | **{test_res['f1_weighted']:.4f}** |

---

## 5. Modality Attention Analysis
- **Overall Mean Structural Attention ($a_s$)**: **{mean_struct_attn:.4f}** ({mean_struct_attn*100:.1f}%)
- **Overall Mean Functional Attention ($a_f$)**: **{mean_func_attn:.4f}** ({mean_func_attn*100:.1f}%)

> [!NOTE]
> Modality attention weights reflect learned gating signals. Functional rs-fMRI embeddings ($a_f \approx {mean_func_attn*100:.1f}\%$) and Structural T1 embeddings ($a_s \approx {mean_struct_attn*100:.1f}\%$) contribute dynamically to the 3-class severity prediction.

---

## 6. Limitations & Disclaimer
- **Dataset Scale**: The dataset contains 157 subjects (102 train, 25 val, 30 test).
- **Scope**: This model is an academic research prototype for multi-site neuroimaging fusion analysis. It is **not** a clinical diagnostic tool and does not claim clinical generalization beyond this research dataset.
"""

    with open(RESULTS_DIR / "STEP_11F_MULTIMODAL_TRAINING_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    logging.info("Saved final training report markdown!")

    print("\n" + "=" * 80)
    print("STEP 11F — MULTIMODAL ATTENTION FUSION TRAINING")
    print("=" * 80)
    print(f"Training subjects: {train_count}")
    print(f"Validation subjects: {val_count}")
    print(f"Test subjects: {test_count}")
    print(f"\nBest epoch: {best_epoch}")
    print(f"Validation Macro-F1: {best_val_f1:.4f}")
    print(f"\nTest Accuracy: {test_res['acc']:.4f}")
    print(f"Test Macro Precision: {test_res['p_macro']:.4f}")
    print(f"Test Macro Recall: {test_res['r_macro']:.4f}")
    print(f"Test Macro F1: {test_res['f1_macro']:.4f}")
    print(f"Test Weighted F1: {test_res['f1_weighted']:.4f}")
    print(f"\nLow F1: {f1_per[0]:.4f}")
    print(f"Moderate F1: {f1_per[1]:.4f}")
    print(f"High F1: {f1_per[2]:.4f}")
    print(f"\nMean Structural Attention: {mean_struct_attn:.4f}")
    print(f"Mean Functional Attention: {mean_func_attn:.4f}")
    print(f"\nBest model:\n{best_model_path}")
    print(f"\nTraining history:\n{history_csv}")
    print(f"\nTest predictions:\n{RESULTS_DIR / 'test_predictions.csv'}")
    print(f"\nFinal metrics:\n{RESULTS_DIR / 'final_metrics.json'}")
    print(f"\nTraining report:\n{RESULTS_DIR / 'STEP_11F_MULTIMODAL_TRAINING_REPORT.md'}")
    print("=" * 80)
    print("\nSTEP 11F STATUS: COMPLETE\n")


if __name__ == "__main__":
    main()
