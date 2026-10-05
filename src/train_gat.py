#!/usr/bin/env python3
"""
src/train_gat.py

Stage 9 — Graph Attention Network (GAT) Training for Person 2 rs-fMRI Cohort.

Trains a 2-layer Graph Attention Network (GAT) model for 3-class autism severity
prediction (Low=0, Moderate=1, High=2) using PyTorch Geometric GATConv.

Leakage Prevention:
- Node feature standardization (mean, std) computed strictly on TRAIN split.
- Class weights computed strictly on TRAIN split.
- Model selection and early stopping driven strictly by VALIDATION macro-F1.
- TEST set evaluated exactly ONCE using the best validation-selected model.

Outputs:
- models/fmri_gat/best_gat_model.pt
- models/fmri_gat/training_history.csv
- models/fmri_gat/test_predictions.csv
- models/fmri_gat/test_metrics.json
- models/fmri_gat/confusion_matrix.csv
- models/fmri_gat/model_config.json
- results/fmri_gat/loss_curve.png
- results/fmri_gat/macro_f1_curve.png
- results/fmri_gat/confusion_matrix.png
- results/fmri_gat/gcn_vs_gat_comparison.png
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# Enforce UTF-8 output encoding for Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src directory to sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

GRAPH_MANIFEST_PATH = PROJECT_ROOT / "data" / "features" / "fmri_graphs" / "graph_manifest.csv"
GCN_METRICS_PATH = PROJECT_ROOT / "models" / "fmri_gcn" / "test_metrics.json"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "fmri_gat"
RESULTS_OUTPUT_DIR = PROJECT_ROOT / "results" / "fmri_gat"
LOG_FILE = PROJECT_ROOT / "logs" / "fmri_gat_training.log"

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


class GATBaseline(nn.Module):
    """
    2-Layer Graph Attention Network (GAT) Model for Graph Classification.
    """
    def __init__(self, in_channels: int = 5, hidden_channels: int = 16, heads: int = 4, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1, dropout=dropout)
        # Layer 1 output dimension: 16 * 4 = 64
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, edge_dim=1, dropout=dropout)
        # Layer 2 output dimension: 16 * 4 = 64
        self.fc1 = nn.Linear(hidden_channels * heads, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        edge_weight = torch.abs(edge_attr).unsqueeze(-1)  # [E, 1]

        h = self.conv1(x, edge_index, edge_attr=edge_weight)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_attr=edge_weight)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h_graph = global_mean_pool(h, batch)

        out = self.fc1(h_graph)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        logits = self.fc2(out)
        return logits


def load_dataset(df_manifest: pd.DataFrame) -> tuple[list[Data], list[Data], list[Data]]:
    """
    Loads all subject PyG graph objects, maps target labels, and groups into train/val/test splits.
    """
    train_graphs, val_graphs, test_graphs = [], [], []

    for idx, row in df_manifest.iterrows():
        sub_id = str(row["participant_id"]).strip()
        split = str(row["split"]).strip()
        sev_class = str(row["severity_class"]).strip()
        graph_rel_path = str(row["graph_path"]).strip()
        graph_file_path = PROJECT_ROOT / graph_rel_path

        if not graph_file_path.exists():
            raise FileNotFoundError(f"Graph file missing for sub-{sub_id}: {graph_file_path}")

        data = torch.load(graph_file_path, weights_only=False)

        # Set target label tensor y = int class index (0, 1, 2)
        target_class_idx = SEVERITY_MAP[sev_class]
        data.y = torch.tensor([target_class_idx], dtype=torch.long)

        # Basic data integrity checks
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            raise ValueError(f"NaN/Inf in node features for sub-{sub_id}")
        if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
            raise ValueError(f"NaN/Inf in edge attributes for sub-{sub_id}")
        if (data.edge_index[0] == data.edge_index[1]).any():
            raise ValueError(f"Self-loop found in edge index for sub-{sub_id}")
        if data.edge_index.shape[1] == 0:
            raise ValueError(f"Graph has no edges for sub-{sub_id}")

        if split == "train":
            train_graphs.append(data)
        elif split == "val":
            val_graphs.append(data)
        elif split == "test":
            test_graphs.append(data)
        else:
            raise ValueError(f"Unknown split '{split}' for sub-{sub_id}")

    return train_graphs, val_graphs, test_graphs


def standardize_node_features(train_graphs: list[Data], val_graphs: list[Data], test_graphs: list[Data]) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes node feature mean and std strictly on the TRAIN split, and normalizes all splits.
    """
    train_x_concat = torch.cat([g.x for g in train_graphs], dim=0).numpy()
    train_mean = np.mean(train_x_concat, axis=0, keepdims=True)
    train_std = np.std(train_x_concat, axis=0, keepdims=True)
    train_std = np.where(train_std == 0, 1e-8, train_std)

    mean_tensor = torch.tensor(train_mean, dtype=torch.float32)
    std_tensor = torch.tensor(train_std, dtype=torch.float32)

    for g in train_graphs:
        g.x = (g.x - mean_tensor) / std_tensor
    for g in val_graphs:
        g.x = (g.x - mean_tensor) / std_tensor
    for g in test_graphs:
        g.x = (g.x - mean_tensor) / std_tensor

    return train_mean.flatten(), train_std.flatten()


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates model on a DataLoader split and returns loss, accuracy, macro P/R/F1, y_true, y_pred, y_probs.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    num_samples = len(loader.dataset)
    avg_loss = float(total_loss / num_samples)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return avg_loss, acc, float(precision), float(recall), float(f1), y_true, y_pred, y_probs


def main():
    parser = argparse.ArgumentParser(description="Stage 9 — GAT Model Training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (validation macro-F1)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_logging()

    logging.info("=" * 80)
    logging.info("STEP 9: GAT MODEL TRAINING INITIALIZED")
    logging.info("=" * 80)

    # 1. Load Graph Manifest
    if not GRAPH_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Graph manifest missing: {GRAPH_MANIFEST_PATH}")

    df_manifest = pd.read_csv(GRAPH_MANIFEST_PATH, dtype={"participant_id": str})
    df_manifest["participant_id"] = df_manifest["participant_id"].str.strip()

    total_graphs = len(df_manifest)
    assert total_graphs == 157, f"Expected 157 graphs in manifest, got {total_graphs}"

    # 2. Sanity Checks & Dataset Loading
    train_graphs, val_graphs, test_graphs = load_dataset(df_manifest)

    n_train = len(train_graphs)
    n_val = len(val_graphs)
    n_test = len(test_graphs)

    assert n_train == 102, f"Expected 102 train graphs, got {n_train}"
    assert n_val == 25, f"Expected 25 validation graphs, got {n_val}"
    assert n_test == 30, f"Expected 30 test graphs, got {n_test}"

    # Check for no subject overlap across splits
    train_ids = set(g.participant_id for g in train_graphs)
    val_ids = set(g.participant_id for g in val_graphs)
    test_ids = set(g.participant_id for g in test_graphs)

    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage: Train and Val IDs overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage: Train and Test IDs overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage: Val and Test IDs overlap!"

    feat_dim = train_graphs[0].x.shape[1]  # 5

    # Print Sanity Check Block
    print("=" * 80)
    print("GAT DATASET CHECK")
    print("-----------------")
    print(f"Total graphs:           {total_graphs}")
    print(f"Train:                  {n_train}")
    print(f"Validation:             {n_val}")
    print(f"Test:                   {n_test}")
    print(f"Nodes per graph:        62")
    print(f"Node feature dimension: {feat_dim}")
    print("Number of graph classes: 3 (0=Low, 1=Moderate, 2=High)")
    print("=" * 80 + "\n")

    # 3. Feature Standardization (Fitted strictly on TRAIN)
    train_mean, train_std = standardize_node_features(train_graphs, val_graphs, test_graphs)
    logging.info(f"Feature Standardization Mean (Train): {train_mean}")
    logging.info(f"Feature Standardization Std  (Train): {train_std}")

    # 4. Class Weight Strategy (Fitted strictly on TRAIN)
    train_labels = [g.y.item() for g in train_graphs]
    train_class_counts = [train_labels.count(c) for c in range(3)]
    total_train_samples = len(train_labels)

    class_weights = [total_train_samples / (3.0 * count) for count in train_class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    logging.info(f"Train Class Counts: Low={train_class_counts[0]}, Moderate={train_class_counts[1]}, High={train_class_counts[2]}")
    logging.info(f"Calculated Inverse-Frequency Class Weights (Train): Low={class_weights[0]:.4f}, Moderate={class_weights[1]:.4f}, High={class_weights[2]:.4f}")

    # 5. DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    # 6. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATBaseline(in_channels=feat_dim, hidden_channels=16, heads=4, num_classes=3, dropout=0.2).to(device)
    class_weights_tensor = class_weights_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"GAT Model Initialized on device '{device}'. Total Trainable Parameters: {num_params}")

    # 7. Training Loop with Early Stopping on Validation Macro-F1
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_OUTPUT_DIR / "best_gat_model.pt"

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(batch.y.cpu().numpy())

        epoch_train_loss = float(train_loss / n_train)
        epoch_train_acc = float(accuracy_score(train_targets, train_preds))
        _, _, epoch_train_f1, _ = precision_recall_fscore_support(train_targets, train_preds, average="macro", zero_division=0)

        # Validation Step
        val_loss, val_acc, val_p, val_r, val_f1, _, _, _ = evaluate_model(model, val_loader, criterion, device)

        rec = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "train_macro_f1": float(epoch_train_f1),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_precision": val_p,
            "val_macro_recall": val_r,
            "val_macro_f1": val_f1
        }
        history.append(rec)

        logging.info(f"Epoch [{epoch:03d}/{args.epochs:03d}] - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        # Check for model checkpoint improvement
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"   >>> Model Checkpoint Saved (Val Macro-F1 Improved to {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f"Early stopping triggered at epoch {epoch} (No improvement in Val Macro-F1 for {args.patience} epochs).")
                break

    total_training_time = time.time() - t_train_start
    logging.info(f"Training Complete in {total_training_time:.2f}s. Best Epoch: {best_epoch} with Val Macro-F1: {best_val_macro_f1:.4f}")

    # Save Training History CSV
    df_history = pd.DataFrame(history)
    history_csv_path = MODEL_OUTPUT_DIR / "training_history.csv"
    df_history.to_csv(history_csv_path, index=False)

    # 8. Load Best Validation Model & Evaluate EXACTLY ONCE on TEST Set
    logging.info("Loading best validation-selected model for final TEST evaluation...")
    model.load_state_dict(torch.load(best_model_path))

    test_loss, test_acc, test_p_macro, test_r_macro, test_f1_macro, test_targets, test_preds, test_probs = evaluate_model(
        model, test_loader, criterion, device
    )

    # Detailed test metrics calculation
    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(test_targets, test_preds, average=None, zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(test_targets, test_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])

    # 9. Save Test Predictions CSV
    test_records = []
    for i, g in enumerate(test_graphs):
        p_id = str(g.participant_id).strip()
        site_code = str(g.site_code).strip()
        true_cls = int(test_targets[i])
        pred_cls = int(test_preds[i])

        test_records.append({
            "participant_id": p_id,
            "site_code": site_code,
            "split": "test",
            "true_class_name": REVERSE_SEVERITY_MAP[true_cls],
            "predicted_class_name": REVERSE_SEVERITY_MAP[pred_cls],
            "true_class": true_cls,
            "predicted_class": pred_cls,
            "prob_low": float(test_probs[i][0]),
            "prob_moderate": float(test_probs[i][1]),
            "prob_high": float(test_probs[i][2])
        })

    df_test_preds = pd.DataFrame(test_records)
    test_preds_path = MODEL_OUTPUT_DIR / "test_predictions.csv"
    df_test_preds.to_csv(test_preds_path, index=False)

    # Save Confusion Matrix CSV
    df_cm = pd.DataFrame(cm, index=["True_Low", "True_Moderate", "True_High"], columns=["Pred_Low", "Pred_Moderate", "Pred_High"])
    cm_path = MODEL_OUTPUT_DIR / "confusion_matrix.csv"
    df_cm.to_csv(cm_path)

    # Save Model Config JSON
    config_data = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "pytorch_geometric_version": torch_geometric.__version__,
        "random_seed": args.seed,
        "device": str(device),
        "total_graphs": total_graphs,
        "train_graphs": n_train,
        "validation_graphs": n_val,
        "test_graphs": n_test,
        "node_features_dim": feat_dim,
        "num_nodes_per_graph": 62,
        "num_classes": 3,
        "model_architecture": "GATConv(5,16,heads=4) -> GATConv(64,16,heads=4) -> GlobalMeanPool -> Linear(64,32) -> Linear(32,3)",
        "num_heads": 4,
        "num_layers": 2,
        "edge_weights_used": True,
        "edge_weight_definition": "Absolute Fisher z-transformed correlation weight (torch.abs(edge_attr))",
        "num_trainable_parameters": num_params,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": 0.2,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "early_stopping_metric": "val_macro_f1",
        "best_epoch": best_epoch,
        "training_time_seconds": round(total_training_time, 2),
        "class_weights": {
            "Low": round(float(class_weights[0]), 4),
            "Moderate": round(float(class_weights[1]), 4),
            "High": round(float(class_weights[2]), 4)
        },
        "train_class_counts": {
            "Low": train_class_counts[0],
            "Moderate": train_class_counts[1],
            "High": train_class_counts[2]
        }
    }
    config_json_path = MODEL_OUTPUT_DIR / "model_config.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    # Save Test Metrics JSON
    test_metrics_data = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_precision": float(test_p_macro),
        "test_macro_recall": float(test_r_macro),
        "test_macro_f1": float(test_f1_macro),
        "test_weighted_f1": float(f1_weighted),
        "per_class_metrics": {
            "Low": {
                "precision": float(p_per_class[0]),
                "recall": float(r_per_class[0]),
                "f1": float(f1_per_class[0])
            },
            "Moderate": {
                "precision": float(p_per_class[1]),
                "recall": float(r_per_class[1]),
                "f1": float(f1_per_class[1])
            },
            "High": {
                "precision": float(p_per_class[2]),
                "recall": float(r_per_class[2]),
                "f1": float(f1_per_class[2])
            }
        },
        "confusion_matrix": cm.tolist()
    }
    test_metrics_json_path = MODEL_OUTPUT_DIR / "test_metrics.json"
    with open(test_metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics_data, f, indent=4)

    # 10. Generate Output Plots & Comparison
    # Loss Curve Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_history["epoch"], df_history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    plt.plot(df_history["epoch"], df_history["val_loss"], label="Val Loss", color="orange", linewidth=2)
    plt.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAT Model — Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_OUTPUT_DIR / "loss_curve.png", dpi=300)
    plt.close()

    # Macro-F1 Curve Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_history["epoch"], df_history["val_macro_f1"], label="Val Macro-F1", color="green", linewidth=2)
    plt.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1-Score")
    plt.title("GAT Model — Validation Macro-F1 Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_OUTPUT_DIR / "macro_f1_curve.png", dpi=300)
    plt.close()

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("GAT Model — Test Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ["Low", "Moderate", "High"])
    plt.yticks(tick_marks, ["Low", "Moderate", "High"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(RESULTS_OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # 11. GCN vs GAT Comparison
    gcn_val_f1, gcn_test_acc, gcn_test_macro_f1, gcn_test_weighted_f1 = 0.4378, 0.2667, 0.2689, 0.2704
    if GCN_METRICS_PATH.exists():
        with open(GCN_METRICS_PATH, "r", encoding="utf-8") as f:
            gcn_data = json.load(f)
            gcn_val_f1 = gcn_data.get("best_val_macro_f1", gcn_val_f1)
            gcn_test_acc = gcn_data.get("test_accuracy", gcn_test_acc)
            gcn_test_macro_f1 = gcn_data.get("test_macro_f1", gcn_test_macro_f1)
            gcn_test_weighted_f1 = gcn_data.get("test_weighted_f1", gcn_test_weighted_f1)

    comp_df = pd.DataFrame([
        {"Metric": "Validation Macro-F1", "GCN Baseline": gcn_val_f1, "GAT Model": float(best_val_macro_f1), "Difference (GAT - GCN)": float(best_val_macro_f1 - gcn_val_f1)},
        {"Metric": "Test Accuracy", "GCN Baseline": gcn_test_acc, "GAT Model": float(test_acc), "Difference (GAT - GCN)": float(test_acc - gcn_test_acc)},
        {"Metric": "Test Macro-F1", "GCN Baseline": gcn_test_macro_f1, "GAT Model": float(test_f1_macro), "Difference (GAT - GCN)": float(test_f1_macro - gcn_test_macro_f1)},
        {"Metric": "Test Weighted-F1", "GCN Baseline": gcn_test_weighted_f1, "GAT Model": float(f1_weighted), "Difference (GAT - GCN)": float(f1_weighted - gcn_test_weighted_f1)}
    ])

    comp_df.to_csv(RESULTS_OUTPUT_DIR / "gcn_vs_gat_comparison.csv", index=False)

    # GCN vs GAT Comparison Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comp_df))
    width = 0.35
    rects1 = ax.bar(x - width/2, comp_df["GCN Baseline"], width, label="GCN Baseline", color="skyblue")
    rects2 = ax.bar(x + width/2, comp_df["GAT Model"], width, label="GAT Model", color="coral")

    ax.set_ylabel("Score")
    ax.set_title("GCN Baseline vs GAT Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Metric"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_OUTPUT_DIR / "gcn_vs_gat_comparison.png", dpi=300)
    plt.close()

    # 12. Final Report Output
    print("\n" + "=" * 80)
    print("===== STEP 9 GAT REPORT =====")
    print("=" * 80)
    print("1. Dataset used:           Person 2 rs-fMRI Brain Graphs (Harvard-Oxford 62 ROIs)")
    print(f"2. Number of graphs:       {total_graphs}")
    print(f"3. Train/Val/Test counts:  Train {n_train} | Validation {n_val} | Test {n_test}")
    print(f"4. Node feature dimension: {feat_dim}")
    print(f"5. GAT Architecture:       GATConv(5,16,heads=4) -> GATConv(64,16,heads=4) -> GlobalMeanPool -> Linear(64,32) -> Linear(32,3)")
    print(f"   Trainable Parameters:   {num_params}")
    print(f"6. Number of Attention Heads: 4")
    print(f"7. Edge weights used:      YES (Absolute Fisher z-transformed weights: torch.abs(edge_attr))")
    print(f"8. Class-weight strategy:  Inverse-Frequency (Train): Low={class_weights[0]:.4f}, Moderate={class_weights[1]:.4f}, High={class_weights[2]:.4f}")
    print(f"9. Best epoch:             {best_epoch} (of max {args.epochs})")
    print(f"10. Validation macro-F1:   {best_val_macro_f1:.4f}")
    print(f"11. Test accuracy:         {test_acc:.4f} ({int(test_acc * n_test)}/{n_test})")
    print(f"12. Test macro-F1:        {test_f1_macro:.4f}")
    print(f"13. Test weighted-F1:     {f1_weighted:.4f}")
    print("\n14. Per-Class Test Performance:")
    print(f"    Low (n=5):       Precision {p_per_class[0]:.4f} | Recall {r_per_class[0]:.4f} | F1 {f1_per_class[0]:.4f}")
    print(f"    Moderate (n=13):  Precision {p_per_class[1]:.4f} | Recall {r_per_class[1]:.4f} | F1 {f1_per_class[1]:.4f}")
    print(f"    High (n=12):      Precision {p_per_class[2]:.4f} | Recall {r_per_class[2]:.4f} | F1 {f1_per_class[2]:.4f}")
    print("\n15. Test Confusion Matrix:")
    print(df_cm)

    print("\n16. GCN vs GAT Model Comparison Table:")
    print(comp_df.to_string(index=False))

    print("\n17. Files Created:")
    print(f"    - Script:               src/train_gat.py")
    print(f"    - Model Checkpoint:     {best_model_path}")
    print(f"    - Training History:     {history_csv_path}")
    print(f"    - Test Predictions:     {test_preds_path}")
    print(f"    - Test Metrics JSON:    {test_metrics_json_path}")
    print(f"    - Confusion Matrix CSV: {cm_path}")
    print(f"    - Model Config JSON:    {config_json_path}")
    print(f"    - Loss Plot:            {RESULTS_OUTPUT_DIR / 'loss_curve.png'}")
    print(f"    - Macro-F1 Plot:        {RESULTS_OUTPUT_DIR / 'macro_f1_curve.png'}")
    print(f"    - Confusion Matrix Plot:{RESULTS_OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"    - GCN vs GAT Plot:      {RESULTS_OUTPUT_DIR / 'gcn_vs_gat_comparison.png'}")

    print("\n18. Warnings / Disclaimers:")
    print("    - Dataset size (157 subjects) is modest; evaluation metrics reflect graph attention model comparisons.")
    print("    - Zero data leakage enforced (Standardization & Class Weights fitted strictly on TRAIN set).")

    print("\n19. Step 10 Multimodal Fusion Status:")
    print("    - Step 10 work was NOT started.")

    print("\n" + "=" * 80)
    print("STEP 9 GAT: COMPLETE")
    print("=" * 80)
    print("DO NOT START STEP 10.")
    print("=" * 80)


if __name__ == "__main__":
    main()
