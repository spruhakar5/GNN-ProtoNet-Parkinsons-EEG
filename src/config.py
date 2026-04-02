"""
Configuration for GNN-ProtoNet: EEG-based Parkinson's Detection.
"""

import os
from pathlib import Path
import torch

# ── Project paths ──
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── EEG parameters ──
# Canonical 32-channel template (10-20 system) — same as MCPNet
COMMON_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2',
    'F9', 'F10',
]
N_CHANNELS = len(COMMON_CHANNELS)  # 32

# Frequency bands for PSD and PLV
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50),
}
N_BANDS = len(FREQ_BANDS)

# ── Preprocessing ──
TARGET_SFREQ = 500       # Resample all datasets to this
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 50.0
NOTCH_FREQS = [50.0, 60.0]
EPOCH_DURATION = 1.0      # seconds

# ── Dataset-specific info ──
DATASETS = {
    'UC': {
        'openneuro_id': 'ds003490',
        'sfreq': 512,
        'pd_subjects': 15,
        'hc_subjects': 16,
    },
    'UNM': {
        'openneuro_id': 'ds002778',
        'sfreq': 500,
        'pd_subjects': 14,
        'hc_subjects': 14,
    },
    'Iowa': {
        'openneuro_id': 'ds004584',
        'sfreq': 500,
        'pd_subjects': 14,
        'hc_subjects': 14,
    },
}

# ── Node features ──
# PSD (5) + time-domain (4: mean,std,skew,kurt) + Hjorth (3) + sample entropy (1) = 13
N_NODE_FEATURES = 13

# ── Graph construction ──
TOP_K = 8                 # top-k neighbors for graph sparsification
USE_PLV_EDGE_ATTR = False # if True, store per-band PLV as edge attributes

# ── GNN encoder ──
ENCODER_TYPE = 'gat'      # 'gat' or 'gcn'
GAT_HEADS = [4, 4, 1]     # attention heads per layer
GAT_HIDDEN = 64           # hidden dim per head
GCN_HIDDEN = 256          # hidden dim for GCN layers
N_GNN_LAYERS = 3
DROPOUT = 0.3
EMBEDDING_DIM = 128

# ── Few-shot settings ──
N_WAY = 2                 # PD vs HC
K_SHOTS = [1, 5, 10, 20]
N_QUERY = 15              # query samples per class per episode
N_EPISODES_TRAIN = 100
N_EPISODES_TEST = 50
CALIBRATION_ALPHA = 0.5

# ── Training ──
LEARNING_RATE = 1e-3
N_TRAIN_EPOCHS = 50
SCHEDULER_STEP = 20
SCHEDULER_GAMMA = 0.5

# ── Device ──
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # MPS not yet supported by PyG scatter ops
