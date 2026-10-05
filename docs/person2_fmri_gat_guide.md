# Person 2 Developer Guide: rs-fMRI & Graph Attention Network (GAT) Pipeline

**Target Audience**: Person 2 (rs-fMRI + GAT Pipeline Developer)  
**Project**: ABIDE II Multimodal Functional-Structural Fusion for ASD Classification  
**Repository Path**: `mri 2/`  
**Shared Data Helper**: [`src/shared_data_utils.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/shared_data_utils.py)  
**Master Split Manifest**: [`data/phenotypic/split_master_manifest.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/split_master_manifest.csv)  
**Site Scanner Reference**: [`data/phenotypic/site_scanner_reference.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/site_scanner_reference.csv)

---

## 1. Overview & Pipeline Hand-off Objectives

As Person 2, your role is to build the **resting-state functional MRI (rs-fMRI) Graph Attention Network (GAT)** pipeline. To ensure seamless fusion with Person 1's structural 3D ResNet-18 pipeline, both pipelines **MUST** adhere to strict dataset boundaries, shared subject splits, and compatible embedding output specifications.

### Key Hand-off Mandates
1. **Identical Subject Splits**: Do **NOT** generate a random train/val/test split. Load subjects strictly using [`src/shared_data_utils.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/shared_data_utils.py) or [`data/phenotypic/split_master_manifest.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/split_master_manifest.csv).
2. **Dual-Modality Cohort**: Focus on the **975 subjects** flagged with `has_dual_modality == True` (32 subjects lack raw rs-fMRI scans on S3).
3. **Site-Dependent TR Bandpass Filtering**: Use site-specific Repetition Time (TR) values from [`data/phenotypic/site_scanner_reference.csv`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/data/phenotypic/site_scanner_reference.csv) to compute Nyquist frequencies.
4. **512-Dimensional Output Embedding**: Ensure your GAT model produces a **512-dimensional graph embedding vector** ($h_{func} \in \mathbb{R}^{B \times 512}$) to align with Person 1's 3D ResNet structural embedding vector ($h_{struct} \in \mathbb{R}^{B \times 512}$).

---

## 2. Quick-Start Data Ingestion (Python API)

You can load all split files, subject IDs, and scanner metadata directly using the provided utility functions in [`src/shared_data_utils.py`](file:///c:/Users/ridhi%20malawade/OneDrive/Desktop/mri%202/src/shared_data_utils.py):

```python
import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))

import shared_data_utils as sdu

# 1. Load site scanner metadata (TR, scanner model, recommended bandpass cutoffs)
df_scanners = sdu.load_site_scanner_info()
print(df_scanners[["site_code", "tr_seconds", "recommended_bandpass_max_hz"]])

# 2. Load Train, Val, and Test splits for the Dual-Modality cohort (975 subjects)
train_df = sdu.load_split("train", cohort="dual_modality")
val_df   = sdu.load_split("val",   cohort="dual_modality")
test_df  = sdu.load_split("test",  cohort="dual_modality")

print(f"Dual-Modality Cohort -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 3. Format S3 URI for a subject's raw rs-fMRI BOLD scan
sample_sub = train_df.iloc[0]
s3_func_uri = sdu.get_s3_func_path(sample_sub["site_code"], sample_sub["participant_id"])
print(f"Sample S3 Functional URI: {s3_func_uri}")
```

---

## 3. rs-fMRI Preprocessing Contract

> [!IMPORTANT]  
> **Raw BIDS Access**: rs-fMRI BOLD files are located on S3 at:  
> `s3://fcp-indi/data/Projects/ABIDE2/RawData/ABIDEII-{site_code}/sub-{participant_id}/ses-1/func/sub-{participant_id}_ses-1_task-rest_run-1_bold.nii.gz`

### Preprocessing Workflow Requirements
1. **Brain Extraction & Spatial Registration**:
   - Perform skull-stripping (e.g., via ANTsPy or Nilearn).
   - Register functional volumes to the **MNI152 standard 1mm coordinate template space** (to match Person 1's structural registration).
2. **Confound Regression**:
   - Regress out 24 motion parameters (6 rigid-body motion params, 6 temporal derivatives, and their 12 quadratic terms).
   - Regress out mean White Matter (WM) and Cerebrospinal Fluid (CSF) signals.
3. **TR-Dependent Temporal Bandpass Filtering**:
   - TR varies across sites (from **0.475s** at `ONRC_2` to **3.000s** at `BNI_1` and `UCLA_1`).
   - Calculate Nyquist frequency per site: $f_{Nyquist} = \frac{1}{2 \times TR}$.
   - Apply bandpass filter: $f_{low} = 0.01\text{ Hz}$, $f_{high} = \min(0.10, 0.8 \times f_{Nyquist})\text{ Hz}$.

### Site-by-Site TR Reference Table

| Site Code | Phenotypic Subjects | S3 rs-fMRI Scans | TR (seconds) | Nyquist Limit ($f_{Nyquist}$) | Bandpass Range (Hz) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ONRC_2** | 59 | 59 | **0.475s** | 1.0526 Hz | 0.01 – 0.1000 Hz |
| **IU_1** | 40 | 40 | **0.813s** | 0.6150 Hz | 0.01 – 0.1000 Hz |
| **EMC_1** | 54 | 54 | **2.000s** | 0.2500 Hz | 0.01 – 0.1000 Hz |
| **GU_1** | 106 | 75 | **2.000s** | 0.2500 Hz | 0.01 – 0.1000 Hz |
| **NYU_1** | 78 | 78 | **2.000s** | 0.2500 Hz | 0.01 – 0.1000 Hz |
| **NYU_2** | 27 | 27 | **2.000s** | 0.2500 Hz | 0.01 – 0.1000 Hz |
| **SDSU_1** | 58 | 58 | **2.000s** | 0.2500 Hz | 0.01 – 0.1000 Hz |
| **TCD_1** | 42 | 42 | **2.000s** | 0.2500 Hz | 0.01 – 0.1000 Hz |
| **UCD_1** | 32 | 32 | **2.000s** | 0.2500 Hz | 0.01 – 0.1000 Hz |
| **USM_1** | 33 | 33 | **2.000s** | 0.2500 Hz | 0.01 – 0.1000 Hz |
| **KKI_1** | 211 | 211 | **2.500s** | 0.2000 Hz | 0.01 – 0.1000 Hz |
| **KUL_3** | 28 | 28 | **2.500s** | 0.2000 Hz | 0.01 – 0.1000 Hz |
| **OHSU_1** | 93 | 93 | **2.500s** | 0.2000 Hz | 0.01 – 0.1000 Hz |
| **IP_1** | 56 | 55 | **2.700s** | 0.1852 Hz | 0.01 – 0.1000 Hz |
| **BNI_1** | 58 | 58 | **3.000s** | 0.1667 Hz | 0.01 – 0.1000 Hz |
| **UCLA_1** | 32 | 32 | **3.000s** | 0.1667 Hz | 0.01 – 0.1000 Hz |

---

## 4. Functional Connectivity Graph Construction

1. **Parcellation Atlas Selection**:
   - Extract ROI mean time-series using a standardized functional atlas (e.g. **AAL-116**, **Schaefer-100**, or **CC200**).
   - *Note*: If region-level explainability is required between structural and functional models, use **Harvard-Oxford Cortical + Subcortical** to match Person 1.
2. **Adjacency Matrix Generation**:
   - Compute full functional connectivity (FC) correlation matrix $C \in \mathbb{R}^{N_{nodes} \times N_{nodes}}$ using Pearson's correlation or Partial Correlation (with Fisher $z$-transformation).
   - Apply absolute thresholding (e.g., $|r| > 0.3$) or construct a $k$-Nearest Neighbors ($k$-NN, $k=8$) graph to form edge index tensor `edge_index`.
3. **Node Feature Matrix ($X \in \mathbb{R}^{N_{nodes} \times F}$)**:
   - Option A: Mean ROI time-series statistics (mean, variance, power spectral density).
   - Option B: Full functional connectivity profile (row vector $C_{i, :}$ of length $N_{nodes}$).

---

## 5. PyTorch Geometric (PyG) Data Object Specification

For each subject, construct a `torch_geometric.data.Data` object matching this contract:

```python
import torch
from torch_geometric.data import Data

# Example for AAL-116 atlas (N=116 nodes)
N_NODES = 116
F_NODE_FEATS = 116  # FC profile per node

node_features = torch.randn(N_NODES, F_NODE_FEATS, dtype=torch.float32)  # Shape: [116, 116]
edge_index    = torch.randint(0, N_NODES, (2, 450), dtype=torch.long)    # Shape: [2, E]
edge_attr     = torch.randn(450, 1, dtype=torch.float32)                  # Shape: [E, 1]
label         = torch.tensor([1], dtype=torch.long)                       # 1 = ASD, 0 = Control

graph_data = Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=label,
    subject_key="USM_1_29495"
)
```

---

## 6. Graph Attention Network (GAT) Architecture & Embedding Spec

To ensure downstream compatibility with Person 1's 3D ResNet structural pipeline, your GAT model **MUST** include a projection layer outputting a **512-dimensional functional embedding vector** ($h_{func} \in \mathbb{R}^{B \times 512}$).

### PyTorch Model Interface Template

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATFunctionalEncoder(nn.Module):
    def __init__(self, in_features=116, hidden_dim=128, heads=4, out_embed_dim=512, num_classes=2):
        super(GATFunctionalEncoder, self).__init__()
        
        # Layer 1: Multi-head Graph Attention
        self.gat1 = GATConv(in_features, hidden_dim, heads=heads, concat=True)
        
        # Layer 2: Second GAT Layer
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        
        # Linear Projection to match Person 1's 512-dim structural embedding
        self.proj_head = nn.Linear(hidden_dim, out_embed_dim)
        
        # Standalone Classification Head (for single-modality benchmark)
        self.fc_classifier = nn.Linear(out_embed_dim, num_classes)

    def extract_embedding(self, x, edge_index, batch):
        """
        Extracts 512-dimensional graph embedding vector h_func.
        Output shape: [Batch_Size, 512]
        """
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        
        # Global Graph Pooling over nodes
        graph_embed = global_mean_pool(x, batch)  # Shape: [B, hidden_dim]
        
        # Project to 512 dimensions
        h_func = self.proj_head(graph_embed)       # Shape: [B, 512]
        return h_func

    def forward(self, x, edge_index, batch):
        h_func = self.extract_embedding(x, edge_index, batch)
        logits = self.fc_classifier(h_func)
        return logits, h_func
```

---

## 7. Multimodal Fusion Head Interface

Once Person 1's 3D ResNet (`h_struct` $\in \mathbb{R}^{B \times 512}$) and Person 2's GAT (`h_func` $\in \mathbb{R}^{B \times 512}$) are trained, they will be combined in the final fusion module:

```python
class MultimodalFusionClassifier(nn.Module):
    def __init__(self, d_struct=512, d_func=512, num_classes=2):
        super(MultimodalFusionClassifier, self).__init__()
        
        # Cross-Modal Concatenation (512 + 512 = 1024)
        self.fusion_fc = nn.Sequential(
            nn.Linear(d_struct + d_func, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, h_struct, h_func):
        # Concatenate structural and functional embeddings along feature dimension
        h_fused = torch.cat([h_struct, h_func], dim=1)  # Shape: [B, 1024]
        logits = self.fusion_fc(h_fused)               # Shape: [B, 2]
        return logits
```

---

## 8. Person 2 Action Checklist

- [ ] **Step 1**: Import subject list using `src/shared_data_utils.load_split("train", cohort="dual_modality")`.
- [ ] **Step 2**: Download raw BOLD NIfTI files from S3 for the 975 dual-modality subjects.
- [ ] **Step 3**: Preprocess rs-fMRI scans applying site-specific TR bandpass limits from `data/phenotypic/site_scanner_reference.csv`.
- [ ] **Step 4**: Compute functional connectivity matrices and build PyG `Data` objects.
- [ ] **Step 5**: Implement `GATFunctionalEncoder` ensuring `extract_embedding()` returns shape `(B, 512)`.
- [ ] **Step 6**: Evaluate standalone GAT model on held-out test set (`load_split("test", cohort="dual_modality")`) using Accuracy, Balanced Accuracy, and ROC-AUC.
- [ ] **Step 7**: Pass pre-trained GAT model weights to Person 1 for multimodal fusion training.
