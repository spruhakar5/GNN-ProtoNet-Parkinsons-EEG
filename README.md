# GNN-ProtoNet: Graph Neural Network with Prototypical Few-Shot Learning for Early Parkinson's Disease Detection from EEG

A research pipeline for early detection of Parkinson's Disease (PD) from resting-state EEG using Graph Neural Networks (GAT/GCN) with prototypical few-shot learning. Replaces traditional CNN encoders with GNN encoders that model EEG as a brain connectivity graph, preserving spatial electrode relationships and enabling interpretable attention-based analysis.

## Key Features

- **Graph-based EEG representation**: 32-node graphs where nodes = electrodes, edges = PLV functional connectivity
- **GAT encoder** with multi-head attention for learning which brain connections discriminate PD from healthy controls
- **GCN encoder** as ablation baseline
- **MCPNet-style preprocessing**: canonical 32-channel template ensuring cross-dataset consistency
- **Prototypical few-shot classification** with test-subject calibration for small-dataset robustness
- **Two evaluation protocols**: Leave-One-Subject-Out (LOSO) and Cross-Dataset Generalization
- **Publication-quality visualizations**: attention heatmaps on electrode layouts, t-SNE embeddings, prototype distances

## Architecture

```
Raw EEG (3 datasets, 230 subjects)
    |
    v
Preprocessing (bandpass, notch, ICA, 32-channel harmonization, 1s epochs)
    |
    v
Feature Extraction (13-dim per electrode: PSD[5] + time-domain[4] + Hjorth[3] + SampEn[1])
    |
    v
Graph Construction (PLV connectivity, top-k=8 sparsification)
    |
    v
GNN Encoder (GAT: 3 layers, multi-head attention, dual readout -> 128-dim embedding)
    |
    v
Prototypical Classifier (support prototypes + calibration -> Euclidean distance classification)
    |
    v
PD / HC prediction
```

## Datasets

Three public EEG datasets from [OpenNeuro](https://openneuro.org):

| Dataset | OpenNeuro ID | Subjects | Sampling Rate |
|---------|-------------|----------|---------------|
| UC San Diego | [ds003490](https://openneuro.org/datasets/ds003490) | 25 PD + 25 HC | 512 Hz |
| UNM | [ds002778](https://openneuro.org/datasets/ds002778) | ~16 PD + ~15 HC | 500 Hz |
| Iowa | [ds004584](https://openneuro.org/datasets/ds004584) | 100 PD + 49 HC | 500 Hz |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/spruhakar5/GNN-ProtoNet-Parkinsons-EEG.git
cd GNN-ProtoNet-Parkinsons-EEG
pip install -r requirements.txt
```

### 2. Download datasets

```bash
cd src
python download_data.py --dataset all
```

Or run the setup script which does everything:

```bash
chmod +x setup.sh
./setup.sh
```

### 3. Run the pipeline

```bash
cd src

# Quick test with synthetic data
python main.py --n_subjects 6 --k_shot 5 --n_episodes 10 --n_epochs 5

# Real data with LOSO evaluation
python main.py --real --k_shot 5

# Cross-dataset generalization
python main.py --real --cross-dataset --k_shot 5

# Full ablation studies (GAT vs GCN, calibration, etc.)
python main.py --real --ablation --k_shot 5

# Generate publication figures
python main.py --real --k_shot 5 --figures

# Use GCN encoder instead of GAT
python main.py --real --encoder gcn --k_shot 5
```

## Project Structure

```
GNN-ProtoNet-Parkinsons-EEG/
├── src/
│   ├── config.py           # All hyperparameters, paths, 32-channel template
│   ├── download_data.py    # OpenNeuro dataset downloader
│   ├── dataset.py          # Subject dataclass, data loading, synthetic generator
│   ├── preprocessing.py    # MCPNet-style: resample, bandpass, notch, ICA, channel harmonization
│   ├── features.py         # Node features (PSD, Hjorth, entropy) + PLV connectivity
│   ├── graph_builder.py    # Top-k sparsified PyG graph construction
│   ├── models/
│   │   ├── gat_encoder.py  # 3-layer GAT with multi-head attention + dual readout
│   │   ├── gcn_encoder.py  # GCN ablation baseline
│   │   └── proto_net.py    # Encoder-agnostic prototypical classifier with calibration
│   ├── train.py            # Episodic few-shot training loop
│   ├── evaluate.py         # LOSO + cross-dataset evaluation protocols
│   ├── visualize.py        # Attention heatmaps, t-SNE, prototype distances, loss curves
│   └── main.py             # Full pipeline runner with CLI
├── data/
│   ├── raw/                # Downloaded OpenNeuro datasets (not tracked)
│   └── processed/          # Cached features and graphs (not tracked)
├── results/                # JSON results and figures (not tracked)
├── docs/                   # Design specs
├── requirements.txt
├── setup.sh
└── README.md
```

## Pipeline Details

### Preprocessing (MCPNet-style)

1. **Resample** to 500 Hz (common rate across datasets)
2. **Bandpass filter**: 0.5-50 Hz (FIR)
3. **Notch filter**: 50/60 Hz power line removal
4. **ICA**: Automatic artifact removal (EOG via frontal channels, kurtosis fallback)
5. **Channel harmonization**: Map to canonical 32-channel 10-20 template
   - Select matching channels, drop extras
   - Interpolate missing channels from spatial neighbors
   - Reorder to fixed canonical order
6. **Epoch**: 1-second non-overlapping windows

### Node Features (13-dim per electrode)

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| PSD band power | 5 | Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-50Hz) |
| Time-domain | 4 | Mean, standard deviation, skewness, kurtosis |
| Hjorth parameters | 3 | Activity, mobility, complexity |
| Sample entropy | 1 | Signal complexity/irregularity |

### Graph Construction

- **Nodes**: 32 electrodes in canonical order
- **Edges**: Phase Locking Value (PLV) averaged across 5 frequency bands
- **Sparsification**: Top-k neighbors (k=8) per node
- **Edge weights**: PLV connectivity strength

### GNN Encoder (GAT)

```
GAT Layer 1: 13 -> 64, 4 attention heads  (output: 32 x 256)
    | ELU + Dropout(0.3)
GAT Layer 2: 256 -> 64, 4 attention heads  (output: 32 x 256)
    | ELU + Dropout(0.3)
GAT Layer 3: 256 -> 128, 1 attention head  (output: 32 x 128)
    |
Graph Readout: mean + max pooling          (output: 256)
    |
MLP Head: 256 -> 128                       (final embedding)
```

### Few-Shot Learning

- **N-way**: 2 (PD vs HC)
- **K-shot**: {1, 5, 10, 20}
- **Episodic training**: 100 episodes/epoch, 50 epochs
- **Prototype calibration**: Adapts prototypes to test subject using calibration samples
- **Classification**: Euclidean distance to class prototypes

### Evaluation

**LOSO**: Train on N-1 subjects, test on 1, repeat for all subjects.

**Cross-Dataset**: Train on 2 datasets, test on the third (3 folds):
- UC + UNM -> Iowa
- UC + Iowa -> UNM
- UNM + Iowa -> UC

**Metrics**: Accuracy, Sensitivity, Specificity, F1-score, AUC-ROC

### Ablation Studies

1. GAT vs GCN encoder
2. With vs without PLV edges
3. With vs without prototype calibration
4. Effect of K-shot (1, 5, 10, 20)
5. Effect of top-k sparsification (k=4, 8, 12)

## CLI Options

```
python main.py [OPTIONS]

Data:
  --real                Use real OpenNeuro datasets
  --n_subjects N        Number of synthetic subjects (default: 10)

Model:
  --encoder {gat,gcn}   GNN encoder type (default: gat)
  --k_shot K            K-shot value (default: runs all [1,5,10,20])
  --top_k K             Graph sparsification k (default: 8)

Evaluation:
  --cross-dataset       Run cross-dataset evaluation
  --ablation            Run all ablation studies
  --no-calibration      Disable prototype calibration

Training:
  --n_episodes N        Episodes per training epoch (default: 100)
  --n_epochs N          Training epochs (default: 50)
  --skip-ica            Skip ICA preprocessing (faster)

Output:
  --figures             Generate publication figures
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4
- MNE-Python >= 1.5
- NumPy, SciPy, scikit-learn, matplotlib, seaborn

## References

- **MCPNet**: Qiu et al. (2024) - Multiscale Convolutional Prototype Network for EEG-based PD detection
- **Prototypical Networks**: Snell et al. (2017) - Prototypical Networks for Few-shot Learning
- **GAT**: Velickovic et al. (2018) - Graph Attention Networks
- **PLV**: Lachaux et al. (1999) - Measuring phase synchrony in brain signals

## License

This project is for academic research purposes.
