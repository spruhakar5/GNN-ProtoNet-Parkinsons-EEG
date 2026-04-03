"""
Feature Extraction for GNN-ProtoNet.

Node features (13-dim per electrode per epoch):
  - PSD band power: delta, theta, alpha, beta, gamma (5)
  - Time-domain: mean, std, skewness, kurtosis (4)
  - Hjorth parameters: activity, mobility, complexity (3)
  - Sample entropy (1)

Edge features:
  - PLV connectivity matrix (32x32) averaged across frequency bands
"""

import numpy as np
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.stats import skew, kurtosis

from config import FREQ_BANDS


# ─────────────────────────────────────────────────────────────
# PSD
# ─────────────────────────────────────────────────────────────

def compute_psd(epoch_data, sfreq):
    """
    PSD band power for a single epoch.

    Parameters
    ----------
    epoch_data : (n_channels, n_times)
    sfreq : float

    Returns
    -------
    psd : (n_channels, 5)
    """
    n_channels = epoch_data.shape[0]
    n_bands = len(FREQ_BANDS)
    psd = np.zeros((n_channels, n_bands))

    for ch in range(n_channels):
        nperseg = min(256, epoch_data.shape[1])
        noverlap = min(128, nperseg - 1)
        freqs, pxx = welch(epoch_data[ch], fs=sfreq,
                           nperseg=nperseg, noverlap=noverlap)
        for b_idx, (_, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
            mask = (freqs >= fmin) & (freqs <= fmax)
            if mask.any():
                psd[ch, b_idx] = np.mean(pxx[mask])
    return psd


# ─────────────────────────────────────────────────────────────
# Time-domain statistics
# ─────────────────────────────────────────────────────────────

def compute_time_domain(epoch_data):
    """
    Time-domain statistics per channel.

    Parameters
    ----------
    epoch_data : (n_channels, n_times)

    Returns
    -------
    stats : (n_channels, 4) — mean, std, skewness, kurtosis
    """
    n_channels = epoch_data.shape[0]
    stats = np.zeros((n_channels, 4))
    for ch in range(n_channels):
        sig = epoch_data[ch]
        stats[ch, 0] = np.mean(sig)
        stats[ch, 1] = np.std(sig)
        stats[ch, 2] = skew(sig)
        stats[ch, 3] = kurtosis(sig)
    return stats


# ─────────────────────────────────────────────────────────────
# Hjorth parameters
# ─────────────────────────────────────────────────────────────

def compute_hjorth(epoch_data):
    """
    Hjorth parameters (activity, mobility, complexity) per channel.

    Parameters
    ----------
    epoch_data : (n_channels, n_times)

    Returns
    -------
    hjorth : (n_channels, 3)
    """
    n_channels = epoch_data.shape[0]
    hjorth = np.zeros((n_channels, 3))
    for ch in range(n_channels):
        sig = epoch_data[ch]
        d1 = np.diff(sig)
        d2 = np.diff(d1)

        activity = np.var(sig)
        mobility_sig = np.sqrt(np.var(d1) / (activity + 1e-10))
        mobility_d1 = np.sqrt(np.var(d2) / (np.var(d1) + 1e-10))
        complexity = mobility_d1 / (mobility_sig + 1e-10)

        hjorth[ch, 0] = activity
        hjorth[ch, 1] = mobility_sig
        hjorth[ch, 2] = complexity
    return hjorth


# ─────────────────────────────────────────────────────────────
# Sample Entropy
# ─────────────────────────────────────────────────────────────

def _sample_entropy(sig, m=2, r_factor=0.2):
    """
    Compute sample entropy of a 1D signal (vectorized).

    Parameters
    ----------
    sig : 1D array
    m : int — embedding dimension
    r_factor : float — tolerance as fraction of std

    Returns
    -------
    sampen : float
    """
    N = len(sig)
    r = r_factor * np.std(sig)
    if r == 0 or N < m + 2:
        return 0.0

    def _count_matches(template_len):
        templates = np.array([sig[i:i + template_len] for i in range(N - template_len)])
        n = len(templates)
        count = 0
        for i in range(n):
            diffs = np.max(np.abs(templates[i] - templates[i+1:]), axis=1)
            count += np.sum(diffs < r)
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)

    if B == 0:
        return 0.0
    return -np.log(A / B) if A > 0 else 0.0


def compute_sample_entropy(epoch_data, max_samples=100):
    """
    Sample entropy per channel. Downsamples long signals for speed.

    Parameters
    ----------
    epoch_data : (n_channels, n_times)
    max_samples : int — downsample to this length for speed

    Returns
    -------
    se : (n_channels, 1)
    """
    n_channels = epoch_data.shape[0]
    se = np.zeros((n_channels, 1))
    for ch in range(n_channels):
        sig = epoch_data[ch]
        # Downsample for speed if needed
        if len(sig) > max_samples:
            step = len(sig) // max_samples
            sig = sig[::step][:max_samples]
        se[ch, 0] = _sample_entropy(sig)
    return se


# ─────────────────────────────────────────────────────────────
# Combined node features
# ─────────────────────────────────────────────────────────────

def extract_node_features_epoch(epoch_data, sfreq):
    """
    Extract all 13 node features for one epoch.

    Parameters
    ----------
    epoch_data : (n_channels, n_times)
    sfreq : float

    Returns
    -------
    features : (n_channels, 13)
    """
    psd = compute_psd(epoch_data, sfreq)           # (n_ch, 5)
    td = compute_time_domain(epoch_data)            # (n_ch, 4)
    hjorth = compute_hjorth(epoch_data)              # (n_ch, 3)
    se = compute_sample_entropy(epoch_data)          # (n_ch, 1)
    return np.concatenate([psd, td, hjorth, se], axis=1)  # (n_ch, 13)


def extract_node_features_all(epochs_data, sfreq):
    """
    Extract node features for all epochs.

    Parameters
    ----------
    epochs_data : (n_epochs, n_channels, n_times)
    sfreq : float

    Returns
    -------
    features : (n_epochs, n_channels, 13)
    """
    n_epochs = epochs_data.shape[0]
    results = []
    for i in range(n_epochs):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    Node features epoch {i+1}/{n_epochs}...", flush=True)
        results.append(extract_node_features_epoch(epochs_data[i], sfreq))
    return np.stack(results, axis=0)


# ─────────────────────────────────────────────────────────────
# PLV
# ─────────────────────────────────────────────────────────────

def _bandpass(data, sfreq, low, high, order=4):
    """Band-pass Butterworth filter."""
    nyq = sfreq / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_plv_epoch(epoch_data, sfreq):
    """
    Compute PLV across all frequency bands, then average.

    Parameters
    ----------
    epoch_data : (n_channels, n_times)
    sfreq : float

    Returns
    -------
    plv_avg : (n_channels, n_channels) — averaged across bands
    """
    n_ch = epoch_data.shape[0]
    n_bands = len(FREQ_BANDS)
    plv_all = np.zeros((n_ch, n_ch, n_bands))

    for b_idx, (_, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
        filtered = _bandpass(epoch_data, sfreq, fmin, fmax)
        analytic = hilbert(filtered, axis=-1)
        phases = np.angle(analytic)  # (n_ch, n_times)

        # Fully vectorized PLV: compute all pairs at once
        # phase_diff[i,j,:] = phases[i,:] - phases[j,:]
        phase_diff = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]  # (n_ch, n_ch, n_times)
        plv_matrix = np.abs(np.mean(np.exp(1j * phase_diff), axis=2))    # (n_ch, n_ch)
        np.fill_diagonal(plv_matrix, 1.0)
        plv_all[:, :, b_idx] = plv_matrix

    # Average across bands
    plv_avg = np.mean(plv_all, axis=2)
    return plv_avg


def compute_plv_all(epochs_data, sfreq):
    """
    Compute PLV for all epochs.

    Parameters
    ----------
    epochs_data : (n_epochs, n_channels, n_times)
    sfreq : float

    Returns
    -------
    plv : (n_epochs, n_channels, n_channels)
    """
    n_epochs = epochs_data.shape[0]
    n_ch = epochs_data.shape[1]
    plv = np.zeros((n_epochs, n_ch, n_ch))
    for i in range(n_epochs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    PLV epoch {i+1}/{n_epochs}...", flush=True)
        plv[i] = compute_plv_epoch(epochs_data[i], sfreq)
    return plv


# ─────────────────────────────────────────────────────────────
# Subject-level extraction
# ─────────────────────────────────────────────────────────────

def extract_features(subject):
    """Extract node features and PLV for one subject."""
    epochs_data = subject.epochs.get_data()
    sfreq = subject.epochs.info['sfreq']
    sid = subject.subject_id

    print(f"  Features for {sid} ({epochs_data.shape[0]} epochs)...")

    print(f"    Computing node features...")
    subject.node_features = extract_node_features_all(epochs_data, sfreq)
    print(f"    Node features: {subject.node_features.shape}")

    print(f"    Computing PLV...")
    subject.plv_matrix = compute_plv_all(epochs_data, sfreq)
    print(f"    PLV: {subject.plv_matrix.shape}")

    return subject


def extract_features_all(subjects):
    """Extract features for all subjects."""
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION")
    print(f"  Node features: 13-dim (PSD[5] + TD[4] + Hjorth[3] + SampEn[1])")
    print(f"  PLV: {len(FREQ_BANDS)} bands averaged")
    print(f"{'='*60}")

    for subj in subjects:
        try:
            extract_features(subj)
        except Exception as e:
            print(f"  [ERROR] {subj.subject_id}: {e}")
    return subjects
