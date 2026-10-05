import json
import os
import sys
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import nibabel as nib

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ASD MRI ANALYSIS — Multimodal MRI Analysis using Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DARK NAVY SCIENTIFIC THEME (CUSTOM CSS)
# Inspired by Visual Reference Layout (Dark slate/navy, glowing accents)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Bar Styling */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #0f172a;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.5rem;
        border-radius: 8px;
    }
    .top-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .top-header p {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Disclaimer Banner */
    .disclaimer-banner {
        background-color: #1e1b4b;
        border-left: 4px solid #818cf8;
        padding: 0.85rem 1.2rem;
        border-radius: 6px;
        color: #e0e7ff;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* Metric Cards */
    .dark-card {
        background-color: #121829;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .dark-card-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .metric-value-large {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .metric-label-sub {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    /* Pipeline Step Box */
    .pipeline-step {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        font-size: 0.85rem;
        color: #e2e8f0;
    }

    /* Severity Badges */
    .badge-low {
        background-color: #064e3b;
        color: #34d399;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-mod {
        background-color: #78350f;
        color: #fbbf24;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-high {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PATHS AND ARTIFACT LOADING
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_T1_DIR = PROJECT_ROOT / "data" / "raw" / "abide2_t1"
ROI_TS_DIR = PROJECT_ROOT / "data" / "features" / "fmri" / "roi_primary"
ATLAS_META_CSV = PROJECT_ROOT / "data" / "features" / "fmri" / "atlas_metadata.csv"

FUNC_EMB_PATH = PROJECT_ROOT / "results" / "gat_embeddings" / "functional_embeddings.csv"
STRUCT_EMB_PATH = PROJECT_ROOT / "results" / "structural_embeddings_157.csv"
ALIGNED_EMB_PATH = PROJECT_ROOT / "results" / "multimodal" / "final_aligned_embeddings_157.csv"
MANIFEST_PATH = PROJECT_ROOT / "results" / "multimodal" / "final_alignment_manifest.csv"
TEST_PRED_PATH = PROJECT_ROOT / "results" / "multimodal" / "test_predictions.csv"
FINAL_METRICS_PATH = PROJECT_ROOT / "results" / "multimodal" / "final_metrics.json"
CONF_MATRIX_CSV = PROJECT_ROOT / "results" / "multimodal" / "confusion_matrix.csv"
TRAIN_HISTORY_CSV = PROJECT_ROOT / "results" / "multimodal" / "training_history.csv"
GAT_CONNECTIONS_CSV = PROJECT_ROOT / "results" / "explainability" / "gat_important_connections.csv"
GAT_REGIONS_CSV = PROJECT_ROOT / "results" / "explainability" / "gat_important_regions.csv"
ATTN_ANALYSIS_CSV = PROJECT_ROOT / "results" / "multimodal" / "attention_analysis.csv"
GRADCAM_CSV = PROJECT_ROOT / "results" / "gradcam" / "regional_activations_summary.csv"

@st.cache_data
def load_project_artifacts():
    artifacts = {}
    
    if TEST_PRED_PATH.exists():
        artifacts["test_pred_df"] = pd.read_csv(TEST_PRED_PATH)
    else:
        artifacts["test_pred_df"] = pd.DataFrame()
        
    if ALIGNED_EMB_PATH.exists():
        artifacts["aligned_df"] = pd.read_csv(ALIGNED_EMB_PATH)
    else:
        artifacts["aligned_df"] = pd.DataFrame()
        
    if FINAL_METRICS_PATH.exists():
        with open(FINAL_METRICS_PATH, 'r') as f:
            artifacts["metrics_json"] = json.load(f)
    else:
        artifacts["metrics_json"] = {}
        
    if GAT_REGIONS_CSV.exists():
        artifacts["gat_regions_df"] = pd.read_csv(GAT_REGIONS_CSV)
    else:
        artifacts["gat_regions_df"] = pd.DataFrame()
        
    if GAT_CONNECTIONS_CSV.exists():
        artifacts["gat_connections_df"] = pd.read_csv(GAT_CONNECTIONS_CSV)
    else:
        artifacts["gat_connections_df"] = pd.DataFrame()
        
    if ATTN_ANALYSIS_CSV.exists():
        artifacts["attn_analysis_df"] = pd.read_csv(ATTN_ANALYSIS_CSV)
    else:
        artifacts["attn_analysis_df"] = pd.DataFrame()
        
    if GRADCAM_CSV.exists():
        artifacts["gradcam_df"] = pd.read_csv(GRADCAM_CSV)
    else:
        artifacts["gradcam_df"] = pd.DataFrame()

    return artifacts

artifacts = load_project_artifacts()
test_pred_df = artifacts["test_pred_df"]
aligned_df = artifacts["aligned_df"]

# Build DYNAMIC participant lists matching actual project artifacts (ZERO hardcoding)
if not test_pred_df.empty and "participant_id" in test_pred_df.columns:
    TEST_30_PIDS = sorted([str(pid) for pid in test_pred_df["participant_id"].unique()])
else:
    TEST_30_PIDS = []

if not aligned_df.empty and "participant_id" in aligned_df.columns:
    ALL_157_PIDS = sorted([str(pid) for pid in aligned_df["participant_id"].unique()])
else:
    ALL_157_PIDS = TEST_30_PIDS if TEST_30_PIDS else ["N/A"]

# -----------------------------------------------------------------------------
# CENTRAL PARTICIPANT STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "selected_subject_id" not in st.session_state:
    st.session_state["selected_subject_id"] = TEST_30_PIDS[0] if TEST_30_PIDS else ALL_157_PIDS[0]

def update_selected_subject(new_pid):
    st.session_state["selected_subject_id"] = str(new_pid)

# -----------------------------------------------------------------------------
# DYNAMIC HARVARD-OXFORD ATLAS & PARTICIPANT FC MATRIX LOOKUP
# -----------------------------------------------------------------------------
@st.cache_data
def get_roi_names():
    if ATLAS_META_CSV.exists():
        df_atlas = pd.read_csv(ATLAS_META_CSV)
        return df_atlas["roi_name"].tolist()
    return [f"ROI {i+1}" for i in range(62)]

@st.cache_data
def get_subject_fc_matrix(pid):
    """
    Loads authentic 62-ROI time series for participant ID from data/features/fmri/roi_primary/
    and computes authentic 62x62 Fisher-z functional connectivity matrix.
    Returns: (fisher_z_matrix, error_msg)
    """
    pid_str = str(pid)
    if not ROI_TS_DIR.exists():
        return None, f"ROI directory not found: {ROI_TS_DIR}"
        
    matching_files = list(ROI_TS_DIR.glob(f"*_sub-{pid_str}_*.npy"))
    if not matching_files:
        matching_files = list(ROI_TS_DIR.glob(f"*{pid_str}*.npy"))
        
    if not matching_files:
        return None, f"Functional ROI time series unavailable for Subject {pid_str}"
        
    try:
        ts = np.load(matching_files[0])  # shape (T, 62)
        if ts.ndim != 2 or ts.shape[1] != 62:
            return None, f"Invalid ROI timeseries shape for Subject {pid_str}"
            
        corr = np.corrcoef(ts.T)
        corr = np.nan_to_num(corr)
        np.fill_diagonal(corr, 0.0)
        
        corr_clipped = np.clip(corr, -0.999, 0.999)
        fisher_z = np.arctanh(corr_clipped)
        return fisher_z, ""
    except Exception as e:
        return None, f"Error computing FC matrix for Subject {pid_str}: {str(e)}"

# -----------------------------------------------------------------------------
# AUTHENTIC DYNAMIC DATA LOOKUP BY PARTICIPANT ID (STRICT EQUALITY MATCHING)
# -----------------------------------------------------------------------------
def get_subject_data(pid):
    """
    Looks up exact subject metadata, prediction, split, and modality attention
    strictly using string equality on participant_id across artifacts.
    """
    pid_str = str(pid)
    res = {
        "participant_id": pid_str,
        "true_class": "Severity unavailable",
        "predicted_class": "Prediction unavailable",
        "split": "Unknown",
        "structural_attention": 0.4892,  # Overall verified mean
        "functional_attention": 0.5108,   # Overall verified mean
        "has_subject_attention": False,
        "is_test_subject": False,
        "found_in_cohort": False
    }
    
    if not test_pred_df.empty and "participant_id" in test_pred_df.columns:
        match_test = test_pred_df[test_pred_df["participant_id"].astype(str) == pid_str]
        if not match_test.empty:
            row = match_test.iloc[0]
            res["true_class"] = str(row.get("true_class", "Severity unavailable"))
            res["predicted_class"] = str(row.get("predicted_class", "Prediction unavailable"))
            res["split"] = "Held-out Test Set"
            res["structural_attention"] = float(row.get("structural_attention", 0.4892))
            res["functional_attention"] = float(row.get("functional_attention", 0.5108))
            res["has_subject_attention"] = True
            res["is_test_subject"] = True
            res["found_in_cohort"] = True
            return res
            
    if not aligned_df.empty and "participant_id" in aligned_df.columns:
        match_aligned = aligned_df[aligned_df["participant_id"].astype(str) == pid_str]
        if not match_aligned.empty:
            row = match_aligned.iloc[0]
            res["true_class"] = str(row.get("severity_class", "Severity unavailable"))
            raw_split = str(row.get("split", "train/val"))
            res["split"] = f"Cohort ({raw_split.capitalize()})"
            res["predicted_class"] = "Not in held-out test set (Train/Val subject)"
            res["found_in_cohort"] = True
            return res

    return res

# -----------------------------------------------------------------------------
# AUTHENTIC NIFTI MRI VOLUME LOADING & NORMALIZATION (DYNAMIC BY PID)
# -----------------------------------------------------------------------------
@st.cache_data
def load_nifti_volume_by_pid(pid):
    """
    Loads authentic 3D T1 MRI NIfTI volume (.nii.gz) for a given participant ID.
    Performs robust 1st-99th percentile intensity windowing & 0-1 normalization.
    """
    pid_str = str(pid)
    if not RAW_T1_DIR.exists():
        return None, f"Directory not found: {RAW_T1_DIR}", ""
    
    matching_files = list(RAW_T1_DIR.glob(f"*_{pid_str}_raw.nii.gz"))
    if not matching_files:
        matching_files = list(RAW_T1_DIR.glob(f"*{pid_str}*.nii.gz"))
        
    if not matching_files:
        return None, f"MRI volume unavailable for Subject {pid_str}", ""
        
    nii_path = matching_files[0]
    try:
        img = nib.load(nii_path)
        vol = img.get_fdata().astype(np.float32)
        vol = np.nan_to_num(vol)
        
        non_zero = vol[vol > 0]
        if len(non_zero) > 0:
            p1, p99 = np.percentile(non_zero, (1, 99))
        else:
            p1, p99 = np.percentile(vol, (1, 99))
            
        if p99 > p1:
            vol_norm = np.clip((vol - p1) / (p99 - p1), 0, 1)
        else:
            vol_norm = vol
            
        return vol_norm, "", nii_path.name
    except Exception as e:
        return None, f"Error loading NIfTI volume for Subject {pid_str}: {str(e)}", ""

def render_mri_slice(volume, slice_idx, view_plane="Axial", subject_id=""):
    """
    Extracts and renders a high-quality 2D slice from an authentic 3D MRI volume.
    Applies bicubic interpolation, grayscale colormap, black background, and orientation markers.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.8), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19')
    
    if volume is None:
        ax.text(0.5, 0.5, f"MRI volume unavailable\nfor Subject {subject_id}", 
                color='#f87171', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig, 0

    shape = volume.shape  # (X, Y, Z), e.g. (176, 256, 256)
    
    if view_plane == "Axial":
        max_slices = shape[2]
        idx = max(0, min(slice_idx, max_slices - 1))
        raw_slice = volume[:, :, idx]
        slice_img = np.rot90(raw_slice)
        title_plane = "AXIAL (Z-AXIS)"
        label_left, label_right, label_top, label_bottom = "R", "L", "A", "P"
    elif view_plane == "Coronal":
        max_slices = shape[1]
        idx = max(0, min(slice_idx, max_slices - 1))
        raw_slice = volume[:, idx, :]
        slice_img = np.rot90(raw_slice)
        title_plane = "CORONAL (Y-AXIS)"
        label_left, label_right, label_top, label_bottom = "R", "L", "S", "I"
    else:  # Sagittal
        max_slices = shape[0]
        idx = max(0, min(slice_idx, max_slices - 1))
        raw_slice = volume[idx, :, :]
        slice_img = np.rot90(raw_slice)
        title_plane = "SAGITTAL (X-AXIS)"
        label_left, label_right, label_top, label_bottom = "A", "P", "S", "I"

    ax.imshow(slice_img, cmap='gray', vmin=0, vmax=1, interpolation='bicubic', aspect='equal')
    
    h, w = slice_img.shape
    ax.text(w * 0.04, h * 0.5, label_left, color='#38bdf8', fontsize=11, fontweight='bold', va='center', ha='center')
    ax.text(w * 0.96, h * 0.5, label_right, color='#38bdf8', fontsize=11, fontweight='bold', va='center', ha='center')
    ax.text(w * 0.5, h * 0.04, label_top, color='#38bdf8', fontsize=11, fontweight='bold', va='center', ha='center')
    ax.text(w * 0.5, h * 0.96, label_bottom, color='#38bdf8', fontsize=11, fontweight='bold', va='center', ha='center')
    
    ax.set_title(f"{title_plane} — Slice {idx+1} / {max_slices}\nSubject: {subject_id}", 
                 color='#38bdf8', fontsize=9, pad=10)
    ax.axis('off')
    plt.tight_layout()
    return fig, max_slices

def render_connectivity_heatmap(selected_sub_id="N/A"):
    """
    Renders 62-node Harvard-Oxford Functional Connectivity Heatmap for the selected participant ID.
    If participant FC matrix is available, renders authentic FC matrix.
    """
    fig, ax = plt.subplots(figsize=(5, 4.2), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19')
    
    fc_matrix, err = get_subject_fc_matrix(selected_sub_id)
    roi_names = get_roi_names()
    
    if fc_matrix is not None:
        key_indices = [0, 1, 2, 3, 4, 11, 14, 53, 58, 60]  # Sample 10 key ROIs for readability
        sub_matrix = fc_matrix[np.ix_(key_indices, key_indices)]
        key_labels = [roi_names[i].split(',')[0][:12] for i in key_indices]
        
        im = ax.imshow(sub_matrix, cmap='YlOrRd', vmin=-1.0, vmax=1.5)
        ax.set_xticks(range(len(key_labels)))
        ax.set_yticks(range(len(key_labels)))
        ax.set_xticklabels(key_labels, rotation=45, ha='right', color='#94a3b8', fontsize=8)
        ax.set_yticklabels(key_labels, color='#94a3b8', fontsize=8)
        ax.set_title(f"Fisher-z FC Matrix (Subject: {selected_sub_id})", color='#38bdf8', fontsize=9.5)
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='#94a3b8')
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#94a3b8')
        plt.tight_layout()
        return fig
    else:
        ax.text(0.5, 0.5, f"FC Matrix unavailable\nfor Subject {selected_sub_id}", color='#f87171', ha='center', va='center', fontsize=9)
        ax.axis('off')
        plt.tight_layout()
        return fig

def render_brain_network_graph(selected_sub_id="N/A"):
    """
    Renders 2D Brain Connectivity Network Graph for the selected participant ID.
    Priority 1: Saved GAT Attention Coefficients (if present in gat_important_connections.csv)
    Priority 2: Participant's authentic 62x62 Fisher-z Functional Connectivity Matrix (top 8 connections)
    Fallback: Explicit error message if neither data source exists.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.5), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19')
    
    selected_sub_id_str = str(selected_sub_id)
    gat_conn_df = artifacts.get("gat_connections_df", pd.DataFrame())
    
    # Priority 1: GAT Attention Edges
    sub_gat = pd.DataFrame()
    if not gat_conn_df.empty and "participant_id" in gat_conn_df.columns:
        sub_gat = gat_conn_df[gat_conn_df["participant_id"].astype(str) == selected_sub_id_str]
        
    if not sub_gat.empty:
        unique_rois = list(dict.fromkeys(sub_gat["source_roi_name"].tolist() + sub_gat["target_roi_name"].tolist()))[:8]
        n_nodes = len(unique_rois)
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        nodes = {roi: (np.cos(ang), np.sin(ang)) for roi, ang in zip(unique_rois, angles)}
        
        for name, (nx, ny) in nodes.items():
            ax.scatter(nx, ny, s=320, color='#3b82f6', edgecolors='#38bdf8', linewidth=1.5, zorder=3)
            short_name = name.split(",")[0][:14]
            ax.text(nx*1.24, ny*1.24, short_name, color='#f8fafc', fontsize=7, ha='center', va='center', fontweight='bold')
            
        for _, row in sub_gat.head(8).iterrows():
            src = row["source_roi_name"]
            tgt = row["target_roi_name"]
            w = float(row.get("gat_attention_coefficient", 0.3))
            if src in nodes and tgt in nodes:
                x1, y1 = nodes[src]
                x2, y2 = nodes[tgt]
                ax.plot([x1, x2], [y1, y2], color='#f59e0b', alpha=0.85, linewidth=w*6.0, zorder=2)
                
        ax.set_xlim(-1.65, 1.65)
        ax.set_ylim(-1.65, 1.65)
        ax.axis('off')
        ax.set_title(f"GAT Attention Graph — Subject: {selected_sub_id_str}", color='#38bdf8', fontsize=10, pad=8)
        plt.tight_layout()
        return fig
        
    # Priority 2: Authentic Participant FC Matrix
    fc_matrix, err_fc = get_subject_fc_matrix(selected_sub_id_str)
    roi_names = get_roi_names()
    
    if fc_matrix is not None and len(roi_names) == 62:
        triu_idx = np.triu_indices(62, k=1)
        weights = fc_matrix[triu_idx]
        top_indices = np.argsort(np.abs(weights))[-8:][::-1]
        
        top_rois_idx = set()
        for idx in top_indices:
            top_rois_idx.add(triu_idx[0][idx])
            top_rois_idx.add(triu_idx[1][idx])
        top_rois_list = sorted(list(top_rois_idx))
        
        n_nodes = len(top_rois_list)
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        nodes = {r_idx: (np.cos(ang), np.sin(ang)) for r_idx, ang in zip(top_rois_list, angles)}
        
        for r_idx, (nx, ny) in nodes.items():
            ax.scatter(nx, ny, s=320, color='#0284c7', edgecolors='#38bdf8', linewidth=1.5, zorder=3)
            r_name = roi_names[r_idx].split(",")[0][:14]
            ax.text(nx*1.24, ny*1.24, r_name, color='#f8fafc', fontsize=7, ha='center', va='center', fontweight='bold')
            
        max_abs_w = np.max(np.abs(weights[top_indices])) if len(top_indices) > 0 else 1.0
        for idx in top_indices:
            r1 = triu_idx[0][idx]
            r2 = triu_idx[1][idx]
            w = weights[idx]
            if r1 in nodes and r2 in nodes:
                x1, y1 = nodes[r1]
                x2, y2 = nodes[r2]
                edge_col = '#34d399' if w > 0 else '#f87171'  # Green for positive FC, Red for negative FC
                edge_width = (abs(w) / (max_abs_w + 1e-6)) * 4.0 + 0.8
                ax.plot([x1, x2], [y1, y2], color=edge_col, alpha=0.8, linewidth=edge_width, zorder=2)
                
        ax.set_xlim(-1.65, 1.65)
        ax.set_ylim(-1.65, 1.65)
        ax.axis('off')
        ax.set_title(f"Functional Connectivity Graph — Subject: {selected_sub_id_str}\n(Visualization threshold: Top 8 strongest FC connections)", 
                     color='#38bdf8', fontsize=8.5, pad=8)
        plt.tight_layout()
        return fig
        
    # Fallback: Explicit error notice
    ax.text(0.5, 0.5, f"Participant-level graph data unavailable\nfor Subject {selected_sub_id_str}", 
            color='#f87171', ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & DYNAMIC COHORT SELECTOR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Navigation")
    nav_option = st.radio(
        "Select Workspace View:",
        [
            "🏠 Main Dashboard",
            "🔬 VisualNeuro (Advanced Workspace)",
            "👥 State Viewer (Subject Comparison)",
            "🌐 Functional Connectivity & Graph",
            "🎯 Test Set Predictions (30 Subjects)",
            "📈 Performance & Benchmarks",
            "📊 Dataset & Cohort Analysis",
            "⚠️ Research Disclaimers & Ethics"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📌 Dynamic Subject Selector")
    
    cohort_scope = st.radio(
        "Participant Scope:",
        ["Held-Out Test Set (N=30)", "Full Multimodal Cohort (N=157)"],
        index=0
    )
    
    ACTIVE_PID_LIST = TEST_30_PIDS if cohort_scope == "Held-Out Test Set (N=30)" else ALL_157_PIDS
    
    current_sel = st.session_state["selected_subject_id"]
    default_idx = ACTIVE_PID_LIST.index(current_sel) if current_sel in ACTIVE_PID_LIST else 0
    
    chosen_pid = st.selectbox(
        "Active Participant ID:",
        ACTIVE_PID_LIST,
        index=default_idx
    )
    
    if chosen_pid != st.session_state["selected_subject_id"]:
        update_selected_subject(chosen_pid)
    
    active_pid = st.session_state["selected_subject_id"]
    active_data = get_subject_data(active_pid)
    
    st.markdown("---")
    st.markdown(f"**Active Participant:** `{active_pid}`")
    st.markdown(f"**Cohort Split:** `{active_data['split']}`")
    st.markdown(f"**True Severity:** `{active_data['true_class']}`")
    
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; text-align:center;'>
        <p><b>Academic Research Prototype</b></p>
        <p>Not for Clinical Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP HEADER BAR
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="top-header">
    <div>
        <h1>ASD MRI ANALYSIS</h1>
        <p>Multimodal MRI Analysis using Deep Learning (3D ResNet-18 + GAT Graph Attention Network)</p>
    </div>
    <div style="text-align: right;">
        <span style="background-color:#1e293b; color:#38bdf8; border:1px solid #334155; padding:0.4rem 0.8rem; border-radius:6px; font-size:0.85rem; font-weight:600;">
            Active Subject: {active_pid}
        </span>
        <span style="background-color:#064e3b; color:#34d399; padding:0.4rem 0.8rem; border-radius:6px; font-size:0.85rem; font-weight:600; margin-left:0.5rem;">
            {active_data['split']}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Visible Research Disclaimer
st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <b>RESEARCH DISCLAIMER:</b> This application is an academic research prototype for multi-level autism severity modeling using MRI embeddings. It is <b>NOT</b> intended for clinical diagnosis, patient screening, or medical decision-making.
</div>
""", unsafe_allow_html=True)

# Load Active Subject Data and Authentic NIfTI MRI Volume
true_class = active_data["true_class"]
pred_class = active_data["predicted_class"]
struct_attn = active_data["structural_attention"]
func_attn = active_data["functional_attention"]

active_volume, load_err, nii_filename = load_nifti_volume_by_pid(active_pid)

# -----------------------------------------------------------------------------
# VIEW 1: MAIN DASHBOARD
# -----------------------------------------------------------------------------
if nav_option == "🏠 Main Dashboard":
    
    # 3 Top Modality Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">🧠 Structural MRI</div>
            <p style="font-size:0.85rem; color:#94a3b8;">Analyses brain anatomy and structural patterns using 3D ResNet-18.</p>
            <div style="font-size:1.1rem; font-weight:700; color:#f8fafc;">512-D Structural Embedding</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">🌐 Functional MRI</div>
            <p style="font-size:0.85rem; color:#94a3b8;">Models brain connectivity (62 ROIs) using Graph Attention Network (GAT).</p>
            <div style="font-size:1.1rem; font-weight:700; color:#f8fafc;">512-D Functional Embedding</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">🏗️ Multimodal AI</div>
            <p style="font-size:0.85rem; color:#94a3b8;">Combines structural and functional modality for severity prediction.</p>
            <div style="font-size:1.1rem; font-weight:700; color:#f8fafc;">1024-D Combined Vector</div>
        </div>
        """, unsafe_allow_html=True)

    # Main Grid: Prediction & Visual Viewers
    grid_col1, grid_col2 = st.columns([1.2, 2])
    
    with grid_col1:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown(f"### Predicted Severity (Subject: {active_pid})")
        
        if pred_class.upper() == "LOW":
            badge_html = '<span class="badge-low">LOW</span>'
        elif pred_class.upper() == "MODERATE":
            badge_html = '<span class="badge-mod">MODERATE</span>'
        elif pred_class.upper() == "HIGH":
            badge_html = '<span class="badge-high">HIGH</span>'
        else:
            badge_html = f'<span style="color:#94a3b8;">{pred_class}</span>'
            
        st.markdown(f"""
        <div style="margin: 1rem 0; text-align: center;">
            <div style="font-size:0.9rem; color:#94a3b8; margin-bottom:0.4rem;">Research-Defined Severity Class</div>
            <div style="font-size:1.8rem; font-weight:800; margin-bottom:0.5rem;">{badge_html}</div>
            <div style="font-size:0.85rem; color:#94a3b8;">Ground Truth Class: <b>{true_class}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Softmax Confidence Score")
        st.info("Softmax Confidence: **N/A** (Only final class predictions are exported in `test_predictions.csv`).")
        
        st.markdown("#### Modality Attention Split")
        st.progress(struct_attn)
        if active_data["has_subject_attention"]:
            st.caption(f"Subject Modality Weight — Structural: **{struct_attn*100:.2f}%** | Functional: **{func_attn*100:.2f}%**")
        else:
            st.caption(f"Mean Modality Contribution — Structural: **{struct_attn*100:.2f}%** | Functional: **{func_attn*100:.2f}%**")
        st.markdown("</div>", unsafe_allow_html=True)

    with grid_col2:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown(f"### 🖥️ Authentic MRI Viewer (Subject: {active_pid})")
        if load_err:
            st.warning(f"⚠️ {load_err}")
        else:
            st.caption(f"Source Volume: `{nii_filename}` | Dimensions: `{active_volume.shape[0]}×{active_volume.shape[1]}×{active_volume.shape[2]}`")
            
        plane_tab1, plane_tab2, plane_tab3 = st.tabs(["Axial", "Coronal", "Sagittal"])
        
        with plane_tab1:
            max_z = active_volume.shape[2] if active_volume is not None else 128
            slice_z = st.slider("Axial Slice Index", 1, max_z, max_z // 2, key="slider_axial")
            fig_ax, _ = render_mri_slice(active_volume, slice_z - 1, "Axial", active_pid)
            st.pyplot(fig_ax)
            
        with plane_tab2:
            max_y = active_volume.shape[1] if active_volume is not None else 128
            slice_y = st.slider("Coronal Slice Index", 1, max_y, max_y // 2, key="slider_coronal")
            fig_cor, _ = render_mri_slice(active_volume, slice_y - 1, "Coronal", active_pid)
            st.pyplot(fig_cor)
            
        with plane_tab3:
            max_x = active_volume.shape[0] if active_volume is not None else 128
            slice_x = st.slider("Sagittal Slice Index", 1, max_x, max_x // 2, key="slider_sagittal")
            fig_sag, _ = render_mri_slice(active_volume, slice_x - 1, "Sagittal", active_pid)
            st.pyplot(fig_sag)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Secondary Grid: Brain Analysis & Connectivity Graph
    bg_col1, bg_col2, bg_col3 = st.columns([1, 1, 1])
    
    with bg_col1:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown("### 🧠 Brain Structural Analysis")
        st.caption("Grad-CAM Activation Summary (Top Regions)")
        gradcam_df = artifacts["gradcam_df"]
        if not gradcam_df.empty:
            st.dataframe(gradcam_df.head(6), use_container_width=True, hide_index=True)
        else:
            st.write("Grad-CAM summary artifact not found.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with bg_col2:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown("### 📊 Functional Connectivity")
        st.pyplot(render_connectivity_heatmap(active_pid))
        st.markdown("</div>", unsafe_allow_html=True)

    with bg_col3:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown(f"### 🕸️ Brain Connectivity Graph ({active_pid})")
        st.pyplot(render_brain_network_graph(active_pid))
        st.markdown("</div>", unsafe_allow_html=True)

    # Model Pipeline Diagram Card
    st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
    st.markdown("### 🔗 End-to-End Model Pipeline Architecture")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    with p1:
        st.markdown("""<div class="pipeline-step"><b>1. T1 Structural MRI</b><br/><span style="color:#94a3b8;">128×128×128 Volume</span></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class="pipeline-step"><b>2. 3D ResNet-18</b><br/><span style="color:#38bdf8;">512-D Embedding</span></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""<div class="pipeline-step"><b>3. fMRI rs-BOLD</b><br/><span style="color:#94a3b8;">62 HO Atlas ROIs</span></div>""", unsafe_allow_html=True)
    with p4:
        st.markdown("""<div class="pipeline-step"><b>4. GAT Graph Net</b><br/><span style="color:#38bdf8;">512-D Embedding</span></div>""", unsafe_allow_html=True)
    with p5:
        st.markdown("""<div class="pipeline-step"><b>5. Attention Fusion</b><br/><span style="color:#fbbf24;">1024-D Vector</span></div>""", unsafe_allow_html=True)
    with p6:
        st.markdown("""<div class="pipeline-step"><b>6. Classifier Head</b><br/><span style="color:#34d399;">3-Level Severity</span></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Model Performance Summary Banner (Verified Project Values)
    st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
    st.markdown("### ⚓ Multimodal Model Test Performance Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown("""<div style="text-align:center;"><div class="metric-label-sub">Test Accuracy</div><div class="metric-value-large" style="color:#38bdf8;">40.00%</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div style="text-align:center;"><div class="metric-label-sub">Macro F1-Score</div><div class="metric-value-large" style="color:#fbbf24;">0.3592</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div style="text-align:center;"><div class="metric-label-sub">Macro Precision</div><div class="metric-value-large" style="color:#e2e8f0;">0.3693</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div style="text-align:center;"><div class="metric-label-sub">Macro Recall</div><div class="metric-value-large" style="color:#e2e8f0;">0.3637</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown("""<div style="text-align:center;"><div class="metric-label-sub">ROC-AUC</div><div class="metric-value-large" style="color:#64748b;">N/A</div></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 2: VISUALNEURO (ADVANCED WORKSPACE)
# Inspired by Screenshot 2
# -----------------------------------------------------------------------------
elif nav_option == "🔬 VisualNeuro (Advanced Workspace)":
    st.markdown("## 🔬 VisualNeuro — Advanced Activation & Region Workspace")
    st.caption("Multi-angle anatomical slice inspection and interactive brain region feature analysis.")
    
    col_ctrl, col_display = st.columns([1, 2.5])
    
    with col_ctrl:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown("#### Visual Controls")
        group_sel = st.selectbox("Group Comparison:", ["All Subjects (N=157)", "Low vs High Severity", "Moderate vs High Severity"])
        p_val = st.slider("P-value threshold:", 0.001, 0.05, 0.05, step=0.005)
        opacity = st.slider("Activation Heatmap Opacity:", 0.1, 1.0, 0.75)
        st.markdown("---")
        st.markdown("#### Brain Atlas Selector")
        atlas_sel = st.selectbox("Brain Atlas:", ["Harvard-Oxford 62 ROIs (Project Atlas)"])
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_display:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown(f"### Authentic Multi-Angle NIfTI Slice Views (Subject: {active_pid})")
        if active_volume is not None:
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                fig_ax, _ = render_mri_slice(active_volume, active_volume.shape[2]//2, "Axial", active_pid)
                st.pyplot(fig_ax)
                st.caption("Top (Axial View)")
            with s_col2:
                fig_cor, _ = render_mri_slice(active_volume, active_volume.shape[1]//2, "Coronal", active_pid)
                st.pyplot(fig_cor)
                st.caption("Front (Coronal View)")
            with s_col3:
                fig_sag, _ = render_mri_slice(active_volume, active_volume.shape[0]//2, "Sagittal", active_pid)
                st.pyplot(fig_sag)
                st.caption("Side (Sagittal View)")
        else:
            st.warning(f"MRI volume unavailable for Subject {active_pid}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Parallel Coordinates & Region Table
    st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
    st.markdown("### 📊 GAT Region Importance Rankings (`gat_important_regions.csv`)")
    gat_regions_df = artifacts["gat_regions_df"]
    if not gat_regions_df.empty:
        st.dataframe(gat_regions_df.head(15), use_container_width=True, hide_index=True)
    else:
        st.write("GAT region importance artifact not found.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 3: STATE VIEWER (DYNAMIC SUBJECT COMPARISON WORKSPACE)
# Inspired by Screenshot 3
# -----------------------------------------------------------------------------
elif nav_option == "👥 State Viewer (Subject Comparison)":
    st.markdown("## 👥 State Viewer — Dynamic Multi-Subject Comparison Workspace")
    st.caption("Select ANY three real participants from the project data for side-by-side comparative inspection.")
    
    st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
    st.markdown("### Select 3 Participants to Compare")
    c_col1, c_col2, c_col3 = st.columns(3)
    
    with c_col1:
        comp_sub1 = st.selectbox("Comparison Subject 1:", ALL_157_PIDS, index=0, key="comp_pid_1")
    with c_col2:
        idx2 = 1 if len(ALL_157_PIDS) > 1 else 0
        comp_sub2 = st.selectbox("Comparison Subject 2:", ALL_157_PIDS, index=idx2, key="comp_pid_2")
    with c_col3:
        idx3 = 2 if len(ALL_157_PIDS) > 2 else 0
        comp_sub3 = st.selectbox("Comparison Subject 3:", ALL_157_PIDS, index=idx3, key="comp_pid_3")
    st.markdown("</div>", unsafe_allow_html=True)
    
    comp_cols = st.columns(3)
    
    for col, sub_id in zip(comp_cols, [comp_sub1, comp_sub2, comp_sub3]):
        sub_info = get_subject_data(sub_id)
        sub_vol, sub_err, _ = load_nifti_volume_by_pid(sub_id)
        
        with col:
            st.markdown(f"""<div class="dark-card">""", unsafe_allow_html=True)
            st.markdown(f"### Subject: {sub_id}")
            st.markdown(f"**Cohort Split:** {sub_info['split']}")
            st.markdown(f"**True Severity:** `{sub_info['true_class']}`")
            st.markdown(f"**Predicted Severity:** `{sub_info['predicted_class']}`")
            st.markdown("---")
            st.markdown("**Modality Weights:**")
            if sub_info["has_subject_attention"]:
                st.markdown(f"- Struct: **{sub_info['structural_attention']*100:.1f}%** | Func: **{sub_info['functional_attention']*100:.1f}%**")
            else:
                st.markdown(f"- Struct: **48.9%** | Func: **51.1%** *(Mean)*")
            st.markdown("---")
            st.markdown("**Authentic MRI Slice:**")
            if sub_vol is not None:
                fig_s, _ = render_mri_slice(sub_vol, sub_vol.shape[2]//2, "Axial", sub_id)
                st.pyplot(fig_s)
            else:
                st.caption(f"⚠️ {sub_err}")
            st.markdown("---")
            st.markdown("**Functional Brain Network Graph:**")
            st.pyplot(render_brain_network_graph(sub_id))
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 4: FUNCTIONAL CONNECTIVITY & GRAPH ANALYSIS
# -----------------------------------------------------------------------------
elif nav_option == "🌐 Functional Connectivity & Graph":
    st.markdown("## 🌐 Functional Connectivity & GAT Graph Analysis")
    st.markdown("Analysis of the 62-node Harvard-Oxford functional connectivity brain graph.")
    
    col_fc1, col_fc2 = st.columns([1, 1])
    with col_fc1:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown(f"### 62-Node Connectivity Heatmap ({active_pid})")
        st.pyplot(render_connectivity_heatmap(active_pid))
        st.markdown("</div>", unsafe_allow_html=True)
    with col_fc2:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown(f"### Functional Brain Network Graph ({active_pid})")
        st.pyplot(render_brain_network_graph(active_pid))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
    st.markdown(f"### 🔗 GAT Important Connections for Subject: {active_pid}")
    gat_conn_df = artifacts["gat_connections_df"]
    if not gat_conn_df.empty and "participant_id" in gat_conn_df.columns:
        match_conn = gat_conn_df[gat_conn_df["participant_id"].astype(str) == str(active_pid)]
        if not match_conn.empty:
            st.dataframe(match_conn, use_container_width=True, hide_index=True)
        else:
            st.info(f"GAT attention coefficients unavailable for Subject {active_pid}. Displaying Functional Connectivity Graph above.")
    else:
        st.write("GAT connections artifact not found.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 5: TEST SET PREDICTIONS (30 SUBJECTS)
# -----------------------------------------------------------------------------
elif nav_option == "🎯 Test Set Predictions (30 Subjects)":
    st.markdown("## 🎯 Held-Out Test Set Predictions (N=30 Subjects)")
    st.caption("Complete breakdown of true vs predicted severity classes and modality attention split.")
    
    if not test_pred_df.empty:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.dataframe(test_pred_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("`results/multimodal/test_predictions.csv` not found.")

# -----------------------------------------------------------------------------
# VIEW 6: PERFORMANCE & BENCHMARKS
# -----------------------------------------------------------------------------
elif nav_option == "📈 Performance & Benchmarks":
    st.markdown("## 📈 Verified Model Performance & Benchmarks")
    st.caption("Strict comparison across GCN Baseline, GAT Functional Model, and Multimodal Attention Fusion.")
    
    st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
    perf_data = {
        "Model Architecture": ["GCN Baseline (fMRI)", "GAT Functional Model (fMRI)", "Multimodal Attention Fusion"],
        "Input Modality": ["fMRI Graph (62 ROIs)", "fMRI Graph (62 ROIs)", "Structural (3D ResNet) + fMRI (GAT)"],
        "Test Accuracy": ["26.67%", "40.00%", "40.00%"],
        "Macro Precision": ["N/A", "0.4273", "0.3693"],
        "Macro Recall": ["N/A", "0.3962", "0.3637"],
        "Macro F1-Score": ["0.2689", "0.3738", "0.3592"],
        "Weighted F1-Score": ["0.2704", "0.4075", "0.4072"]
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_cls1, col_cls2 = st.columns([1, 1])
    with col_cls1:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown("### Class-Wise Multimodal Performance")
        class_data = {
            "Research Severity Class": ["Low", "Moderate", "High"],
            "Precision": [0.1250, 0.4444, 0.5385],
            "Recall": [0.2000, 0.3077, 0.5833],
            "F1-Score": [0.1538, 0.3636, 0.5600]
        }
        st.dataframe(pd.DataFrame(class_data), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cls2:
        st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
        st.markdown("### Modality Contribution Split")
        st.markdown("- **Mean Structural Attention:** 48.92%")
        st.markdown("- **Mean Functional Attention:** 51.08%")
        st.caption("Overall mean calculated across all 30 test subjects.")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 7: DATASET & COHORT ANALYSIS
# -----------------------------------------------------------------------------
elif nav_option == "📊 Dataset & Cohort Analysis":
    st.markdown("## 📊 ABIDE II Dataset & Modeling Cohort Breakdown")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("""<div class="dark-card"><div class="dark-card-header">Full ABIDE II Cohort</div><div class="metric-value-large">1,007</div><div class="metric-label-sub">516 ASD | 491 Controls</div></div>""", unsafe_allow_html=True)
    with col_d2:
        st.markdown("""<div class="dark-card"><div class="dark-card-header">ADOS-2 Severity Cohort</div><div class="metric-value-large">228</div><div class="metric-label-sub">Valid ADOS-2 Severity Scores</div></div>""", unsafe_allow_html=True)
    with col_d3:
        st.markdown("""<div class="dark-card"><div class="dark-card-header">Multimodal Modeling Cohort</div><div class="metric-value-large">157</div><div class="metric-label-sub">102 Train | 25 Val | 30 Test</div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
    st.markdown("### Final Severity Class Distribution (N=157)")
    st.markdown("- **Low Severity Class:** 27 subjects")
    st.markdown("- **Moderate Severity Class:** 66 subjects")
    st.markdown("- **High Severity Class:** 64 subjects")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 8: RESEARCH DISCLAIMERS & ETHICS
# -----------------------------------------------------------------------------
elif nav_option == "⚠️ Research Disclaimers & Ethics":
    st.markdown("## ⚠️ Scientific Disclaimers & Ethical Framing")
    
    st.markdown("""<div class="dark-card">""", unsafe_allow_html=True)
    st.markdown("### 1. Research Prototype Notice")
    st.write("This application is strictly an academic research prototype developed for multi-level autism severity modeling. It is **not** a clinical diagnostic tool and must not be used for patient screening, medical diagnosis, or treatment planning.")
    
    st.markdown("### 2. Explainability & Causation")
    st.write("All visualizations (Grad-CAM maps, GAT graph attention coefficients, and modality attention weights) describe internal artificial neural network behavior. **These visualizations do not establish biological causation, medical pathology, or clinical biomarkers.**")
    
    st.markdown("### 3. Severity Terminology")
    st.write("The labels **Low**, **Moderate**, and **High** used throughout this application represent **Research-Defined Severity Classes** derived from ADOS-2 calibrated severity scores within the ABIDE II dataset. They are **not** equivalent to clinical DSM-5 severity levels or medical diagnoses.")
    st.markdown("</div>", unsafe_allow_html=True)
