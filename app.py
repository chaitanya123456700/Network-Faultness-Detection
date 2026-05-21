# Streamlit Dashboard for Network Fault Detection using GNN

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, classification_report

from models.gat_model import FixedFaultDetectionGAT

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Network Fault Detection using GNN",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🧠 Fault Detection in Communication Networks using Graph Learning")
st.markdown("---")

st.markdown(
    """
This dashboard visualizes the performance of a Graph Attention Network (GAT)
for fault detection in communication networks using the NSL-KDD dataset.
"""
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Dataset Information",
        "Graph Visualization",
        "Model Architecture",
        "Training Metrics",
        "Classification Results",
        "Live Snapshot Analysis"
    ]
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data

def load_processed_data():
    with open('data/timeseries_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open('data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    with open('data/class_weights.pkl', 'rb') as f:
        class_weights = pickle.load(f)

    return train_data, test_data, class_weights


train_data, test_data, class_weights = load_processed_data()

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# ============================================================
# SECTION 1 - OVERVIEW
# ============================================================

if section == "Project Overview":

    st.header("📌 Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Objective")
        st.write(
            """
            Detect faults and anomalous activities in communication networks
            using Graph Neural Networks (GNNs), particularly Graph Attention
            Networks (GAT).
            """
        )

        st.subheader("Pipeline")
        st.markdown(
            """
            1. Data Collection
            2. Graph Construction
            3. GAT-based Learning
            4. Fault Classification
            5. Alert Generation
            """
        )

    with col2:
        st.subheader("Model Information")
        st.write("Model: FixedFaultDetectionGAT")
        st.write("Dataset: NSL-KDD")
        st.write("Graph Type: kNN Graph")
        st.write("Node Features: 38")
        st.write("Classes: 5")

# ============================================================
# SECTION 2 - DATASET
# ============================================================

elif section == "Dataset Information":

    st.header("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Snapshots", len(train_data))

    with col2:
        st.metric("Test Snapshots", len(test_data))

    with col3:
        st.metric("Classes", 5)

    st.subheader("Class Labels")

    class_df = pd.DataFrame({
        'Class ID': [0,1,2,3,4],
        'Class Name': CLASS_NAMES
    })

    st.dataframe(class_df, use_container_width=True)

    st.subheader("Class Weights")

    weights_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Weight': class_weights
    })

    st.bar_chart(weights_df.set_index('Class'))

# ============================================================
# SECTION 3 - GRAPH VISUALIZATION
# ============================================================

elif section == "Graph Visualization":

    st.header("🕸️ Graph Visualization")

    snapshot_id = st.slider(
        "Select Snapshot",
        0,
        len(train_data)-1,
        0
    )

    snapshot = train_data[snapshot_id]

    edge_index = snapshot['edge_index']
    features = snapshot['features']
    labels = snapshot['labels']

    G = nx.Graph()

    for i in range(features.shape[0]):
        G.add_node(i, label=int(labels[i]))

    for i in range(edge_index.shape[1]):
        src = int(edge_index[0][i])
        dst = int(edge_index[1][i])
        G.add_edge(src, dst)

    st.write(f"Nodes: {G.number_of_nodes()}")
    st.write(f"Edges: {G.number_of_edges()}")

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    node_colors = [labels[node] for node in G.nodes()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.Set1,
        node_size=50,
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.3,
        ax=ax
    )

    ax.set_title(f"Snapshot {snapshot_id} Graph Structure")
    ax.axis('off')

    st.pyplot(fig)

# ============================================================
# SECTION 4 - MODEL ARCHITECTURE
# ============================================================

elif section == "Model Architecture":

    st.header("🧠 GAT Model Architecture")

    st.code(
        """
Input Features (38)
        ↓
GAT Layer 1
(hidden=64, heads=4)
        ↓
BatchNorm + ELU + Dropout
        ↓
GAT Layer 2
(hidden=64)
        ↓
Residual Connection
        ↓
Fully Connected Layer
        ↓
Output Layer (5 Classes)
        """
    )

    st.subheader("Fault Classes")

    cols = st.columns(5)

    for i, cls in enumerate(CLASS_NAMES):
        cols[i].metric(f"Class {i}", cls)

# ============================================================
# SECTION 5 - TRAINING METRICS
# ============================================================

elif section == "Training Metrics":

    st.header("📈 Training Metrics")

    try:
        img = Image.open('outputs/training_curves.png')
        st.image(img, caption='Training Curves', use_container_width=True)

    except:
        st.warning("training_curves.png not found")

    st.subheader("Performance Summary")

    metrics_df = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'Weighted F1',
            'Macro F1'
        ],
        'Value': [
            0.7796,
            0.7688,
            0.5802
        ]
    })

    st.dataframe(metrics_df, use_container_width=True)

# ============================================================
# SECTION 6 - CLASSIFICATION RESULTS
# ============================================================

elif section == "Classification Results":

    st.header("🎯 Classification Results")

    results_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'F1 Score': [0.8256, 0.8557, 0.7959, 0.3617, 0.0620]
    })

    st.subheader("Per-Class F1 Scores")

    st.bar_chart(results_df.set_index('Class'))

    st.subheader("Detailed Results")

    st.dataframe(results_df, use_container_width=True)

    st.subheader("Confusion Matrix")

    try:
        img = Image.open('outputs/confusion_matrix.png')
        st.image(img, caption='Confusion Matrix', use_container_width=True)

    except:
        st.warning("confusion_matrix.png not found")

# ============================================================
# SECTION 7 - LIVE SNAPSHOT ANALYSIS
# ============================================================

elif section == "Live Snapshot Analysis":

    st.header("⚡ Live Snapshot Analysis")

    snapshot_id = st.slider(
        "Select Snapshot for Analysis",
        0,
        len(test_data)-1,
        0
    )

    snapshot = test_data[snapshot_id]

    st.subheader("Snapshot Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Nodes", snapshot['features'].shape[0])

    with col2:
        st.metric("Edges", snapshot['edge_index'].shape[1])

    with col3:
        st.metric("Feature Dimensions", snapshot['features'].shape[1])

    st.subheader("Node Labels Distribution")

    labels = snapshot['labels']

    unique, counts = np.unique(labels, return_counts=True)

    distribution = pd.DataFrame({
        'Class': [CLASS_NAMES[i] for i in unique],
        'Count': counts
    })

    st.dataframe(distribution, use_container_width=True)

    st.bar_chart(distribution.set_index('Class'))

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <center>
    <h4>Fault Detection in Communication Networks using Graph Learning</h4>
    <p>Developed using PyTorch Geometric + Streamlit</p>
    </center>
    """,
    unsafe_allow_html=True
)
