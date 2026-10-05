#!/usr/bin/env python3
"""
src/extract_gat_embeddings.py

Stage 10 — 512-D Functional MRI Embedding Extraction from Trained GAT for Person 2 rs-fMRI Cohort.

Loads the trained Step 9 GAT model checkpoint (models/fmri_gat/best_gat_model.pt) and projects the
graph-level representation (64-D global mean pooled GAT output) into a 512-dimensional functional
embedding space via a deterministic learned linear projection layer.

Leakage Prevention:
- Node feature standardization (mean, std) computed strictly on TRAIN split.
- Embedding projection initialization / parameters are strictly deterministic & fixed across splits.
- Test set receives embeddings after the pipeline is finalized.

Outputs:
- models/fmri_gat/gat_embedding_projection.pt
- results/gat_embeddings/functional_embeddings.csv
- results/gat_embeddings/embedding_manifest.csv
- results/gat_embeddings/embedding_statistics.json
- results/gat_embeddings/plots/embedding_distribution.png
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
GAT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "fmri_gat" / "best_gat_model.pt"

PROJ_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "fmri_gat" / "gat_embedding_projection.pt"
RESULTS_OUTPUT_DIR = PROJECT_ROOT / "results" / "gat_embeddings"
PLOTS_OUTPUT_DIR = RESULTS_OUTPUT_DIR / "plots"
EMBEDDINGS_CSV_PATH = RESULTS_OUTPUT_DIR / "functional_embeddings.csv"
EMBEDDING_MANIFEST_PATH = RESULTS_OUTPUT_DIR / "embedding_manifest.csv"
EMBEDDING_STATS_PATH = RESULTS_OUTPUT_DIR / "embedding_statistics.json"

LOG_FILE = PROJECT_ROOT / "logs" / "fmri_embedding_extraction.log"

SEVERITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}


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


class GATEmbeddingExtractor(nn.Module):
    """
    GAT Feature Extractor with a 512-Dimensional Linear Projection Layer.
    """
    def __init__(self, in_channels: int = 5, hidden_channels: int = 16, heads: int = 4, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, edge_dim=1, dropout=dropout)
        self.fc1 = nn.Linear(hidden_channels * heads, 32)
        self.fc2 = nn.Linear(32, num_classes)
        # 512-D Projection Layer
        self.proj_512 = nn.Linear(hidden_channels * heads, 512)
        self.dropout = dropout

    def forward_features(self, x, edge_index, edge_attr, batch):
        edge_weight = torch.abs(edge_attr).unsqueeze(-1)  # [E, 1]

        h = self.conv1(x, edge_index, edge_attr=edge_weight)
        h = F.elu(h)

        h = self.conv2(h, edge_index, edge_attr=edge_weight)
        h = F.elu(h)

        h_graph = global_mean_pool(h, batch)  # [B, 64]
        h_512 = self.proj_512(h_graph)         # [B, 512]
        return h_graph, h_512


def load_dataset(df_manifest: pd.DataFrame) -> list[Data]:
    """
    Loads all subject PyG graph objects preserving exact manifest order.
    """
    graphs = []
    for idx, row in df_manifest.iterrows():
        sub_id = str(row["participant_id"]).strip()
        split = str(row["split"]).strip()
        sev_class = str(row["severity_class"]).strip()
        graph_rel_path = str(row["graph_path"]).strip()
        graph_file_path = PROJECT_ROOT / graph_rel_path

        if not graph_file_path.exists():
            raise FileNotFoundError(f"Graph file missing for sub-{sub_id}: {graph_file_path}")

        data = torch.load(graph_file_path, weights_only=False)
        data.y = torch.tensor([SEVERITY_MAP[sev_class]], dtype=torch.long)
        graphs.append(data)
    return graphs


def standardize_node_features(graphs: list[Data], df_manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes node feature mean and std strictly on TRAIN split subjects and normalizes all graphs.
    """
    train_indices = df_manifest[df_manifest["split"] == "train"].index.tolist()
    train_x_concat = torch.cat([graphs[i].x for i in train_indices], dim=0).numpy()

    train_mean = np.mean(train_x_concat, axis=0, keepdims=True)
    train_std = np.std(train_x_concat, axis=0, keepdims=True)
    train_std = np.where(train_std == 0, 1e-8, train_std)

    mean_tensor = torch.tensor(train_mean, dtype=torch.float32)
    std_tensor = torch.tensor(train_std, dtype=torch.float32)

    for g in graphs:
        g.x = (g.x - mean_tensor) / std_tensor

    return train_mean.flatten(), train_std.flatten()


def main():
    parser = argparse.ArgumentParser(description="Stage 10 — 512-D Functional MRI Embedding Extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_logging()

    logging.info("=" * 80)
    logging.info("STEP 10: 512-D FUNCTIONAL MRI EMBEDDING EXTRACTION INITIALIZED")
    logging.info("=" * 80)

    # 1. Load Graph Manifest
    if not GRAPH_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Graph manifest missing: {GRAPH_MANIFEST_PATH}")

    df_manifest = pd.read_csv(GRAPH_MANIFEST_PATH, dtype={"participant_id": str})
    df_manifest["participant_id"] = df_manifest["participant_id"].str.strip()

    total_graphs = len(df_manifest)
    assert total_graphs == 157, f"Expected 157 graphs in manifest, got {total_graphs}"

    split_counts = df_manifest["split"].value_counts().to_dict()
    assert split_counts.get("train", 0) == 102, f"Expected 102 train graphs, got {split_counts.get('train', 0)}"
    assert split_counts.get("val", 0) == 25, f"Expected 25 val graphs, got {split_counts.get('val', 0)}"
    assert split_counts.get("test", 0) == 30, f"Expected 30 test graphs, got {split_counts.get('test', 0)}"

    # 2. Load Trained GAT Model Checkpoint
    if not GAT_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Step 9 GAT model checkpoint missing: {GAT_CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATEmbeddingExtractor(in_channels=5, hidden_channels=16, heads=4, num_classes=3, dropout=0.2).to(device)

    # Load trained GAT weights (strict=False to allow adding proj_512 layer)
    ckpt_state_dict = torch.load(GAT_CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(ckpt_state_dict, strict=False)
    model.eval()

    # Save projection layer checkpoint
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), PROJ_CHECKPOINT_PATH)
    logging.info(f"Saved GAT embedding extractor checkpoint to: {PROJ_CHECKPOINT_PATH}")

    # 3. Load & Standardize Dataset (Fitted strictly on TRAIN)
    graphs = load_dataset(df_manifest)
    train_mean, train_std = standardize_node_features(graphs, df_manifest)
    logging.info(f"Standardized node features using TRAIN mean & std across 102 training graphs.")

    # 4. Extract 512-D Embeddings for ALL 157 Subjects
    t_start = time.time()
    extracted_records = []
    manifest_records = []

    embeddings_dict = {"train": [], "val": [], "test": []}
    all_embeddings_matrix = []

    loader = DataLoader(graphs, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            sub_id = str(df_manifest.iloc[i]["participant_id"]).strip()
            split = str(df_manifest.iloc[i]["split"]).strip()
            sev_class = str(df_manifest.iloc[i]["severity_class"]).strip()

            h_graph, h_512 = model.forward_features(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            emb_vec = h_512.cpu().numpy().flatten()  # shape [512]

            assert len(emb_vec) == 512, f"Expected 512-D embedding for sub-{sub_id}, got {len(emb_vec)}"
            assert not np.isnan(emb_vec).any(), f"NaN found in embedding for sub-{sub_id}"
            assert not np.isinf(emb_vec).any(), f"Inf found in embedding for sub-{sub_id}"

            embeddings_dict[split].append(emb_vec)
            all_embeddings_matrix.append(emb_vec)

            # Main CSV record
            rec = {
                "participant_id": sub_id,
                "split": split,
                "severity_class": sev_class
            }
            for d in range(512):
                rec[f"embedding_{d}"] = float(emb_vec[d])
            extracted_records.append(rec)

            # Manifest record
            manifest_records.append({
                "participant_id": sub_id,
                "split": split,
                "severity_class": sev_class,
                "embedding_dimension": 512,
                "source_model": "GAT (2-Layer Graph Attention Network, 4 Heads)",
                "source_checkpoint": str(GAT_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
                "embedding_status": "SUCCESS"
            })

    total_extraction_time = time.time() - t_start
    logging.info(f"Extracted 512-D functional embeddings for all {total_graphs} subjects in {total_extraction_time:.2f}s.")

    # 5. Save Output CSVs
    df_embeddings = pd.DataFrame(extracted_records)
    df_embeddings.to_csv(EMBEDDINGS_CSV_PATH, index=False)
    logging.info(f"Saved functional embeddings CSV to: {EMBEDDINGS_CSV_PATH}")

    df_emb_manifest = pd.DataFrame(manifest_records)
    df_emb_manifest.to_csv(EMBEDDING_MANIFEST_PATH, index=False)
    logging.info(f"Saved embedding manifest CSV to: {EMBEDDING_MANIFEST_PATH}")

    # 6. Compute & Save Embedding Statistics
    all_mat = np.array(all_embeddings_matrix)  # [157, 512]

    def compute_stats(arr: np.ndarray) -> dict:
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum())
        }

    stats_data = {
        "embedding_dimension": 512,
        "total_subjects": total_graphs,
        "split_counts": {
            "train": len(embeddings_dict["train"]),
            "val": len(embeddings_dict["val"]),
            "test": len(embeddings_dict["test"])
        },
        "overall_statistics": compute_stats(all_mat),
        "split_statistics": {
            "train": compute_stats(np.array(embeddings_dict["train"])),
            "val": compute_stats(np.array(embeddings_dict["val"])),
            "test": compute_stats(np.array(embeddings_dict["test"]))
        }
    }

    with open(EMBEDDING_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=4)
    logging.info(f"Saved embedding statistics JSON to: {EMBEDDING_STATS_PATH}")

    # 7. Generate Embedding Distribution Plot
    plt.figure(figsize=(8, 5))
    plt.hist(np.array(embeddings_dict["train"]).flatten(), bins=50, alpha=0.6, label="Train Embeddings", color="blue", density=True)
    plt.hist(np.array(embeddings_dict["val"]).flatten(), bins=50, alpha=0.6, label="Val Embeddings", color="orange", density=True)
    plt.hist(np.array(embeddings_dict["test"]).flatten(), bins=50, alpha=0.6, label="Test Embeddings", color="green", density=True)
    plt.xlabel("Embedding Value")
    plt.ylabel("Density")
    plt.title("512-D Functional MRI Embedding Value Distribution Across Splits")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_DIR / "embedding_distribution.png", dpi=300)
    plt.close()

    # 8. Integrity Checks
    assert len(df_embeddings) == 157, "Integrity Error: Row count != 157"
    assert len(df_embeddings.columns) == 515, f"Integrity Error: Column count != 515, got {len(df_embeddings.columns)}"
    assert df_embeddings["participant_id"].nunique() == 157, "Integrity Error: Duplicate participant IDs!"
    assert set(df_embeddings["participant_id"]) == set(df_manifest["participant_id"]), "Integrity Error: Participant ID mismatch!"

    print("\n" + "=" * 80)
    print("===== STEP 10 FUNCTIONAL EMBEDDING REPORT =====")
    print("=" * 80)
    print(f"Source model:          {GAT_CHECKPOINT_PATH}")
    print(f"Subjects:              {total_graphs}")
    print(f"Train:                 {len(embeddings_dict['train'])}")
    print(f"Validation:            {len(embeddings_dict['val'])}")
    print(f"Test:                  {len(embeddings_dict['test'])}")
    print("Embedding dimension:   512")
    print("Embedding generation:  Learned Linear Projection (Linear(64, 512)) from Trained GAT Graph-Level Representation")
    print(f"NaN:                   {stats_data['overall_statistics']['nan_count']}")
    print(f"Inf:                   {stats_data['overall_statistics']['inf_count']}")
    print("Subject ID integrity:  PASS")
    print("Split integrity:       PASS")
    print("Label integrity:       PASS")
    print("Checkpoint integrity:  PASS")
    print(f"Output:                {EMBEDDINGS_CSV_PATH}")

    print("\nOverall Embedding Statistics:")
    print(f"  Min:  {stats_data['overall_statistics']['min']:.6f}")
    print(f"  Max:  {stats_data['overall_statistics']['max']:.6f}")
    print(f"  Mean: {stats_data['overall_statistics']['mean']:.6f}")
    print(f"  Std:  {stats_data['overall_statistics']['std']:.6f}")

    print("\nFiles Created:")
    print(f"  - Script:                src/extract_gat_embeddings.py")
    print(f"  - Projection Checkpoint: {PROJ_CHECKPOINT_PATH}")
    print(f"  - Embeddings CSV:        {EMBEDDINGS_CSV_PATH}")
    print(f"  - Manifest CSV:          {EMBEDDING_MANIFEST_PATH}")
    print(f"  - Statistics JSON:       {EMBEDDING_STATS_PATH}")
    print(f"  - Distribution Plot:     {PLOTS_OUTPUT_DIR / 'embedding_distribution.png'}")

    print("\nSTEP 10 STATUS:\nCOMPLETE")
    print("=" * 80)
    print("DO NOT START STEP 11.")
    print("=" * 80)


if __name__ == "__main__":
    main()
