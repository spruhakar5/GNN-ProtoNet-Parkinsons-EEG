"""
Dataset loading for GNN-ProtoNet.

Supports three OpenNeuro EEG datasets:
1. UC San Diego (ds003490) - 15 PD + 16 HC
2. UNM (ds002778) - 14 PD + 14 HC
3. Iowa (ds004584) - 14 PD + 14 HC
"""

import os
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import mne

from config import DATA_RAW, DATA_PROCESSED, DATASETS, COMMON_CHANNELS, N_CHANNELS, TARGET_SFREQ


@dataclass
class Subject:
    """Represents a single EEG subject."""
    subject_id: str
    dataset: str          # 'UC', 'UNM', or 'Iowa'
    label: int            # 0 = HC, 1 = PD
    raw_path: str = ""
    raw: Optional[mne.io.BaseRaw] = field(default=None, repr=False)
    epochs: Optional[mne.Epochs] = field(default=None, repr=False)
    # Node features: (n_epochs, n_channels, n_node_features)
    node_features: Optional[np.ndarray] = field(default=None, repr=False)
    # PLV connectivity: (n_epochs, n_channels, n_channels)
    plv_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    # PyG graph list
    graphs: Optional[list] = field(default=None, repr=False)


def load_participants_tsv(dataset_dir: Path, dataset_name: str = '') -> dict:
    """
    Load participants.tsv -> {subject_id: label} mapping.
    Handles different TSV formats across datasets:
      - UC: column 'Group' with values 'PD' / 'CTL'
      - UNM: no group column, label inferred from subject ID (sub-hc* / sub-pd*)
      - Iowa: column 'GROUP' with values 'PD' / 'Control'
    """
    tsv_path = dataset_dir / "participants.tsv"
    labels = {}
    if not tsv_path.exists():
        print(f"  [WARN] No participants.tsv at {tsv_path}")
        return labels

    with open(tsv_path, 'r') as f:
        header = f.readline().strip().split('\t')
        header_lower = [h.lower() for h in header]
        id_col = header_lower.index('participant_id')

        # Find group column (may be 'group', 'Group', 'GROUP', or absent)
        group_col = None
        for i, h in enumerate(header_lower):
            if h == 'group':
                group_col = i
                break

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) <= id_col:
                continue
            sid = parts[id_col].strip()

            if group_col is not None and len(parts) > group_col:
                group = parts[group_col].upper().strip()
                # PD label: 'PD'
                # HC label: 'CTL', 'CONTROL', 'HC', 'HEALTHY', or anything else
                labels[sid] = 1 if group == 'PD' else 0
            else:
                # UNM-style: infer from subject ID
                sid_lower = sid.lower()
                if 'pd' in sid_lower:
                    labels[sid] = 1
                elif 'hc' in sid_lower or 'ctl' in sid_lower or 'control' in sid_lower:
                    labels[sid] = 0
                # else: skip unknown

    n_pd = sum(1 for v in labels.values() if v == 1)
    n_hc = sum(1 for v in labels.values() if v == 0)
    print(f"  Loaded {len(labels)} labels ({n_pd} PD, {n_hc} HC) from {tsv_path}")
    return labels


def discover_eeg_files(dataset_dir: Path) -> list:
    """Find all EEG files in a dataset directory."""
    patterns = ['**/*.set', '**/*.edf', '**/*.bdf', '**/*.fif', '**/*.vhdr']
    files = []
    for pat in patterns:
        files.extend(dataset_dir.glob(pat))
    return sorted(files)


def load_raw_eeg(filepath: Path, dataset_name: str) -> mne.io.BaseRaw:
    """Load a single raw EEG file using MNE."""
    ext = filepath.suffix.lower()
    if ext == '.set':
        raw = mne.io.read_raw_eeglab(str(filepath), preload=True, verbose=False)
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
    elif ext == '.bdf':
        raw = mne.io.read_raw_bdf(str(filepath), preload=True, verbose=False)
    elif ext == '.fif':
        raw = mne.io.read_raw_fif(str(filepath), preload=True, verbose=False)
    elif ext == '.vhdr':
        raw = mne.io.read_raw_brainvision(str(filepath), preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return raw


def load_dataset(dataset_name: str) -> List[Subject]:
    """Load all subjects from one dataset."""
    dataset_dir = DATA_RAW / dataset_name
    if not dataset_dir.exists():
        print(f"[SKIP] Not found: {dataset_dir}")
        return []

    print(f"\nLoading dataset: {dataset_name}")
    labels = load_participants_tsv(dataset_dir, dataset_name)
    eeg_files = discover_eeg_files(dataset_dir)
    if not eeg_files:
        print(f"  No EEG files found in {dataset_dir}")
        return []

    print(f"  Found {len(eeg_files)} EEG files")

    # For datasets with multiple sessions (UC), use only session 1
    # to avoid duplicate subjects
    seen_subjects = set()
    subjects = []
    for fpath in eeg_files:
        sub_id = None
        for p in fpath.parts:
            if p.startswith('sub-'):
                sub_id = p
                break
        if sub_id is None:
            sub_id = fpath.stem

        # Skip duplicate sessions — keep first encountered
        if sub_id in seen_subjects:
            continue
        seen_subjects.add(sub_id)

        label = labels.get(sub_id, -1)
        if label == -1:
            continue

        print(f"  Loading {sub_id} ({'PD' if label == 1 else 'HC'})... ", end="", flush=True)
        try:
            raw = load_raw_eeg(fpath, dataset_name)
            subj = Subject(
                subject_id=sub_id,
                dataset=dataset_name,
                label=label,
                raw_path=str(fpath),
                raw=raw,
            )
            subjects.append(subj)
            print(f"OK ({len(raw.ch_names)} ch, {raw.n_times/raw.info['sfreq']:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    n_pd = sum(1 for s in subjects if s.label == 1)
    n_hc = sum(1 for s in subjects if s.label == 0)
    print(f"  Loaded: {len(subjects)} ({n_pd} PD, {n_hc} HC)")
    return subjects


def load_all_datasets() -> List[Subject]:
    """Load all three datasets."""
    all_subjects = []
    for ds_name in ['UC', 'UNM', 'Iowa']:
        all_subjects.extend(load_dataset(ds_name))
    n_pd = sum(1 for s in all_subjects if s.label == 1)
    n_hc = sum(1 for s in all_subjects if s.label == 0)
    print(f"\nTOTAL: {len(all_subjects)} subjects ({n_pd} PD, {n_hc} HC)")
    return all_subjects


def generate_synthetic_data(n_subjects=20, sfreq=500, duration_sec=60) -> List[Subject]:
    """
    Generate synthetic EEG data for pipeline testing.
    PD subjects: elevated theta, reduced beta power.
    HC subjects: normal spectral profile.
    """
    print(f"\nGenerating {n_subjects} synthetic subjects ({sfreq} Hz, {duration_sec}s)...")
    subjects = []
    n_pd = n_subjects // 2
    n_channels = N_CHANNELS

    for i in range(n_subjects):
        label = 1 if i < n_pd else 0
        sub_id = f"syn-{i+1:03d}"
        np.random.seed(42 + i)
        n_samples = sfreq * duration_sec
        t = np.arange(n_samples) / sfreq

        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            phase = np.random.rand(5) * 2 * np.pi
            delta = 20 * np.sin(2 * np.pi * 2 * t + phase[0])
            theta = 10 * np.sin(2 * np.pi * 6 * t + phase[1])
            alpha = 8 * np.sin(2 * np.pi * 10 * t + phase[2])
            beta  = 5 * np.sin(2 * np.pi * 20 * t + phase[3])
            gamma = 2 * np.sin(2 * np.pi * 40 * t + phase[4])

            if label == 1:  # PD: more theta, less beta
                theta *= 1.5
                beta *= 0.6
            else:
                theta *= 0.8
                beta *= 1.2

            # Add inter-channel correlation for realistic PLV
            neighbor_signal = 0.3 * np.sin(2 * np.pi * 10 * t + phase[2] + 0.1 * ch)
            noise = 3 * np.random.randn(n_samples)
            data[ch] = delta + theta + alpha + beta + gamma + neighbor_signal + noise

        data *= 1e-6  # scale to microvolts

        ch_names = COMMON_CHANNELS[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)

        subj = Subject(subject_id=sub_id, dataset='synthetic', label=label, raw=raw)
        subjects.append(subj)
        print(f"  {sub_id} ({'PD' if label==1 else 'HC'})")

    print(f"  Total: {n_pd} PD, {n_subjects - n_pd} HC")
    return subjects
