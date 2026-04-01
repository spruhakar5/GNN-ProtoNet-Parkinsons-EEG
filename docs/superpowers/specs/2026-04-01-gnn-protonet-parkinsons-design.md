# GNN-ProtoNet: Graph Neural Network with Prototypical Few-Shot Learning for Early Parkinson's Detection from EEG

## Overview

Replace MCPNet's CNN encoder with a GNN (GAT) encoder while preserving the MCPNet preprocessing pipeline and prototypical few-shot classification framework. The key insight: EEG is inherently a graph (electrodes=nodes, connectivity=edges), and modelling it as such yields better interpretability and cross-dataset generalization than flattening into 2D matrices for CNNs.

**Pipeline:**
```
Raw EEG -> Preprocess (MCPNet-style) -> Feature Extraction -> Graph Construction -> GNN Encoder -> Graph Embedding -> Prototypical Classifier -> PD/HC
```

## Datasets

Three public EEG datasets from OpenNeuro:

| Dataset | OpenNeuro ID | Subjects | Sampling Rate | Duration |
|---------|-------------|----------|---------------|----------|
| UC San Diego | ds003490 | 15 PD + 16 HC | 512 Hz | ~3 min |
| UNM | ds002778 | 14 PD + 14 HC | 500 Hz | ~2 min |
| Iowa | ds004584 | 14 PD + 14 HC | 500 Hz | ~2 min |

Total: ~87 subjects (43 PD, 44 HC).

## Preprocessing (MCPNet-style, reused)

1. **Resample** to common rate (500 Hz)
2. **Bandpass filter**: 0.5-50 Hz (FIR, firwin)
3. **Notch filter**: 50/60 Hz power line removal
4. **ICA**: artifact removal (EOG, muscle) via FastICA, auto-detect using frontal channels, kurtosis fallback
5. **Channel harmonization**: map to canonical 32-channel template (10-20 system)
   - Select matching channels, drop extras
   - Interpolate missing channels from nearest spatial neighbors (MNE built-in)
   - Reorder to fixed canonical order
6. **Epoch**: 1-second non-overlapping windows

**Canonical 32 channels:**
Fp1, Fp2, F7, F3, Fz, F4, F8, FT7, FC3, FCz, FC4, FT8, T7, C3, Cz, C4, T8, TP7, CP3, CPz, CP4, TP8, P7, P3, Pz, P4, P8, O1, Oz, O2, F9, F10

## Graph Construction

Each 1-second epoch becomes one graph G = (V, E, X, W):

### Nodes (32 per graph)
One per canonical electrode, fixed order across all subjects/datasets.

### Node Features (13-dim per node)
Per electrode, per epoch:
- **PSD band power** (5): delta (0.5-4Hz), theta (4-8Hz), alpha (8-13Hz), beta (13-30Hz), gamma (30-50Hz)
- **Time-domain statistics** (4): mean, std, skewness, kurtosis
- **Hjorth parameters** (3): activity, mobility, complexity
- **Sample entropy** (1)

### Edge Construction
1. Compute PLV between all 32x32 channel pairs across 5 frequency bands
2. Average PLV across bands to get single 32x32 connectivity matrix
3. Apply top-k sparsification (k=8): each node keeps only its 8 strongest connections
4. Edge weights = PLV values (weighted, undirected graph)
5. Optional config flag: per-band PLV as edge attributes

### Missing Channels
Interpolated from nearest spatial neighbors before feature extraction (same as MCPNet). Every subject always produces a fixed 32-node graph.

## GNN Encoder

### Primary: GAT (Graph Attention Network)

```
Input: Graph (32 nodes x 13 features, sparse edges with PLV weights)
  |
GAT Layer 1: 13 -> 64, 4 attention heads (output: 32 x 256)
  | ELU + Dropout(0.3)
GAT Layer 2: 256 -> 64, 4 attention heads (output: 32 x 256)
  | ELU + Dropout(0.3)
GAT Layer 3: 256 -> 128, 1 attention head (output: 32 x 128)
  |
Graph Readout: mean pooling + max pooling concatenated (output: 256)
  |
MLP Head: 256 -> 128 (final embedding dim)
```

- 3 GAT layers: sufficient depth for full-brain information propagation (~4 hop diameter)
- Multi-head attention in early layers captures different connectivity patterns
- Dual readout (mean + max pooling): mean captures global brain state, max captures most discriminative regional signal
- Edge weights injected as attention bias
- Final embedding: 128-dim

### Ablation: GCN Encoder
Same architecture shape, GAT layers replaced with GCN layers. Uses PLV as edge weights directly in message-passing aggregation. No learned attention.

### Implementation
PyTorch Geometric (PyG) for all GNN layers and graph utilities.

## Prototypical Few-Shot Learning

### Episodic Training
- N-way: 2 (PD vs HC)
- K-shot: {1, 5, 10, 20}
- Query: 15 per class per episode
- Episodes per epoch: 100
- Training epochs: 50

### Prototype Computation
- Encode all support graphs through GNN -> 128-dim embeddings
- PD prototype = mean of PD support embeddings
- HC prototype = mean of HC support embeddings

### Prototype Calibration (test time)
- Take a few epochs from test subject as calibration samples
- calibrated_prototype = alpha * original_prototype + (1 - alpha) * subject_mean
- alpha = 0.5 (configurable)

### Classification
- Euclidean distance from query embedding to each prototype
- Log-softmax over negative distances
- Loss: negative log-likelihood

### Training Details
- Optimizer: Adam, lr=1e-3
- Scheduler: StepLR, step_size=20, gamma=0.5
- Dropout: 0.3 in GNN layers

## Evaluation Protocols

### Protocol 1: LOSO (Leave-One-Subject-Out)
- For each of ~87 subjects: train on 86, test on 1
- Report per-subject accuracy, then aggregate

### Protocol 2: Cross-Dataset Generalization
- 3 folds:
  - Train UC+UNM -> Test Iowa
  - Train UC+Iowa -> Test UNM
  - Train UNM+Iowa -> Test UC

### Metrics (both protocols)
- Accuracy, Sensitivity (recall), Specificity, F1-score, AUC-ROC
- Per-class confusion matrix
- Mean +/- std across folds
- Paired t-test: GAT vs GCN significance

### Ablation Studies
1. GAT vs GCN encoder
2. With vs without PLV edges (random graph baseline)
3. With vs without prototype calibration
4. Effect of K-shot (1, 5, 10, 20)
5. Effect of top-k sparsification (k=4, 8, 12)

### Visualization (publication figures)
- Attention weight heatmaps on 10-20 electrode layout
- t-SNE of graph embeddings colored by PD/HC
- Prototype distance distributions
- Training loss curves

## Project Structure

```
parkinsons/
├── src/
│   ├── config.py           # Hyperparameters, paths, channel template
│   ├── download_data.py    # OpenNeuro dataset downloader
│   ├── dataset.py          # Subject dataclass, data loading, synthetic generator
│   ├── preprocessing.py    # MCPNet-style preprocessing pipeline
│   ├── features.py         # Node features (PSD, time-domain, Hjorth, entropy)
│   ├── graph_builder.py    # PLV edge construction, top-k sparsification, PyG Data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gat_encoder.py  # GAT encoder (primary)
│   │   ├── gcn_encoder.py  # GCN encoder (ablation)
│   │   └── proto_net.py    # Prototypical classifier (encoder-agnostic)
│   ├── train.py            # Episodic training loop
│   ├── evaluate.py         # LOSO + cross-dataset evaluation
│   ├── visualize.py        # Attention heatmaps, t-SNE, figures
│   └── main.py             # Pipeline runner
├── data/
│   ├── raw/                # Downloaded OpenNeuro datasets
│   └── processed/          # Cached features and graphs
├── results/                # JSON results, figures
├── requirements.txt
└── setup.sh
```

## Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
mne>=1.5.0
torch>=2.0.0
torch-geometric>=2.4.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
```
