# UI/UX and Visual Design Specification (`design.md`)

**Project Title:** 3-Level Autism Severity Detection Using Multimodal MRI  
**Application File:** `app.py` (Streamlit Dashboard)  
**Design Theme:** Dark Navy / Slate Scientific Medical Research Theme  
**Document Version:** 1.0  

---

## 1. Visual Design Direction & Design System

The visual design system of the Streamlit application (`app.py`) is styled as a professional, dark-mode neuroimaging research workstation. It uses a high-contrast dark navy palette, clear card containers, bright cyan/blue accents, and readable scientific graphics.

### 1.1 Color Palette Specifications
| Design Token | Hex Code | Visual Application |
| :--- | :--- | :--- |
| **App Background** | `#0b0f19` | Main viewport canvas background |
| **Header / Sidebar** | `#0f172a` | Top header navigation bar and left sidebar |
| **Card Container** | `#121829` | Main metric cards and chart panels |
| **Card Border** | `#1e293b` | 1px subtle container dividing border |
| **Primary Accent** | `#38bdf8` | Main titles, selected subject tags, cyan highlights |
| **Secondary Accent** | `#3b82f6` | Network graph nodes, primary badges, buttons |
| **Text Primary** | `#f8fafc` | Main headings, metric values, badge text |
| **Text Secondary** | `#94a3b8` | Subtitles, labels, axis labels, captions |
| **Low Severity Badge** | `#064e3b` / `#34d399` | Low research severity class indicator |
| **Moderate Severity Badge**| `#78350f` / `#fbbf24` | Moderate research severity class indicator |
| **High Severity Badge** | `#7f1d1d` / `#f87171` | High research severity class indicator |

---

## 2. Dynamic Central State & Navigation Architecture

All workspaces are connected to a central Streamlit session state variable (`st.session_state["selected_subject_id"]`). Selecting any participant ID from the sidebar quick selector or top header dropdown dynamically updates all 8 workspace views simultaneously.

```
Sidebar Navigation (st.sidebar.radio)
  ├── 🏠 Main Dashboard
  ├── 🔬 VisualNeuro (Advanced Workspace)
  ├── 👥 State Viewer (Subject Comparison)
  ├── 🌐 Functional Connectivity & Graph
  ├── 🎯 Test Set Predictions (30 Subjects)
  ├── 📈 Performance & Benchmarks
  ├── 📊 Dataset & Cohort Analysis
  └── ⚠️ Research Disclaimers & Ethics
```

---

## 3. Workspace Specifications

### 3.1 🏠 Main Dashboard
- **Purpose**: Provides an executive overview of the dual-modality pipeline, the selected participant's predicted severity, modality attention weights, high-resolution MRI slices, FC heatmap, and GAT brain network graph.
- **Key Components**:
  - Top 3 Modality Cards (Structural 3D ResNet, Functional GAT, Multimodal Fusion).
  - Selected Subject Severity Card (Low / Moderate / High badge, ground truth class, softmax confidence note, progress bar for modality attention split).
  - High-Resolution Authentic MRI Viewer (Axial, Coronal, Sagittal slice tabs with index slider).
  - Structural Analysis Card (Grad-CAM top regional activations table).
  - Functional Connectivity Heatmap (10×10 sample key ROI matrix).
  - GAT Brain Network Graph (Top GAT attention edges or FC matrix edges for selected participant).
  - Pipeline Diagram Flow (6-step card sequence).
  - Model Performance Summary Footer (Accuracy: 40.00%, Macro F1: 0.3592).

### 3.2 🔬 VisualNeuro (Advanced Workspace)
- **Purpose**: Enables deep multi-angle anatomical slice inspection and interactive brain region feature analysis.
- **Key Components**:
  - Control Sidebar (Group comparison selector, P-value slider, opacity slider).
  - Multi-Angle NIfTI Slice Viewports (Simultaneous rendering of Axial, Coronal, and Sagittal orthogonal slices for the selected participant).
  - GAT Region Importance Table (`gat_important_regions.csv` centrality rankings).

### 3.3 👥 State Viewer (Dynamic Subject Comparison Workspace)
- **Purpose**: Enables side-by-side comparative analysis of any three real participants selected dynamically by the user.
- **Key Components**:
  - 3 Dynamic Participant Dropdown Selectors (`Comparison Subject 1`, `Comparison Subject 2`, `Comparison Subject 3`).
  - 3 Column Comparison Panels:
    - Subject Phenotype Card (ID, cohort split, true severity, predicted severity).
    - Modality Attention Split indicator.
    - Authentic Axial MRI Slice viewport for that specific participant.
    - Dynamic Brain Network Graph for that specific participant.

### 3.4 🌐 Functional Connectivity & Graph Analysis
- **Purpose**: Provides in-depth parcellation analysis of the 62 Harvard-Oxford brain regions.
- **Key Components**:
  - Interactive 62-ROI Functional Connectivity Heatmap.
  - Dynamic Brain Network Graph for the active participant.
  - GAT Important Connections Table (`gat_important_connections.csv` edge rankings).

### 3.5 🎯 Test Set Predictions (30 Subjects) Workspace
- **Purpose**: Displays full held-out test set predictions and modality attention weights across all 30 test participants.
- **Key Components**:
  - Interactive DataFrame table rendering `results/multimodal/test_predictions.csv` with sorting and search controls.

### 3.6 📈 Performance & Benchmarks Workspace
- **Purpose**: Presents rigorous comparative benchmark evaluations between single-modality models and multimodal fusion.
- **Key Components**:
  - Model Architecture Comparison Table (GCN vs GAT vs Multimodal Fusion).
  - Class-Wise Performance Metrics Table (Low, Moderate, High Precision, Recall, F1).
  - Modality Contribution Summary Card (Structural: 48.92%, Functional: 51.08%).

### 3.7 📊 Dataset & Cohort Analysis Workspace
- **Purpose**: Details phenotypic breakdown and cohort tier definitions across ABIDE II.
- **Key Components**:
  - Metric summary cards for Full ABIDE II (1,007), ADOS Cohort (228), and Multimodal Cohort (157).
  - Severity Class Distribution summary (Low = 27, Moderate = 66, High = 64).

### 3.8 ⚠️ Research Disclaimers & Ethics Workspace
- **Purpose**: Clearly communicates academic research limitations, explainability boundaries, and clinical disclaimers.
- **Key Components**:
  - Academic Prototype Notice box.
  - Explainability & Causation Disclaimer.
  - Severity Class Terminology clarification.

---

## 4. Component UI Specifications

### 4.1 High-Resolution MRI Viewer Component
- **Data Source**: Authentic 3D T1 NIfTI files (`data/raw/abide2_t1/*_{pid}_raw.nii.gz`).
- **Rendering Pipeline**:
  - 1st–99th percentile intensity windowing on non-zero brain voxels.
  - Orthogonal 2D slice extraction (`vol[:, :, idx]` Axial, `vol[:, idx, :]` Coronal, `vol[idx, :, :]` Sagittal).
  - High-quality bicubic interpolation (`interpolation="bicubic"`, `cmap="gray"`, `aspect="equal"`).
  - Perimeter anatomical orientation labels (`L`, `R`, `A`, `P`, `S`, `I`).
  - Dark background (`#0b0f19`) with no pixelation blocks or distorted aspect ratios.

### 4.2 Brain Network Graph Component
- **Data Source**:
  - *Priority 1*: Saved GAT attention coefficients from `gat_important_connections.csv` (when available).
  - *Priority 2*: Participant's authentic 62×62 Fisher-z Functional Connectivity matrix computed from `data/features/fmri/roi_primary/*_sub-{pid}_roi_timeseries.npy`.
- **Rendering Pipeline**:
  - Circular layout of Harvard-Oxford ROI nodes.
  - Node color: Bright blue (`#3b82f6`) with cyan border (`#38bdf8`).
  - Edge colors: Amber (`#f59e0b`) for GAT attention; Green (`#34d399`) for positive FC; Red (`#f87171`) for negative FC.
  - Edge thickness proportional to connectivity magnitude.
  - Clear visualization threshold label (`Top 8 strongest connections shown`).

### 4.3 Unavailable & Error State Standards
- **Missing MRI File**: Displays yellow warning banner `⚠️ MRI volume unavailable for Subject {pid}`.
- **Missing GAT Attention**: Displays informative banner `GAT attention coefficients unavailable for Subject {pid}. Displaying Subject Functional Connectivity Graph.`
- **Missing Test Prediction**: Displays status `Not in held-out test set (Train/Val subject)`.
