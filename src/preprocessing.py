"""
EEG Preprocessing Pipeline (MCPNet-style).

Steps:
1. Resample to common rate (500 Hz)
2. Band-pass filter (0.5-50 Hz)
3. Notch filter (50/60 Hz)
4. ICA artifact removal
5. Channel harmonization to 32-channel canonical template
6. Epoch segmentation (1s non-overlapping)
"""

import numpy as np
import mne
from mne.preprocessing import ICA

from config import (
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQS,
    COMMON_CHANNELS, EPOCH_DURATION, TARGET_SFREQ,
)


def resample(raw, target_sfreq=TARGET_SFREQ):
    """Resample to common sampling rate if needed."""
    if raw.info['sfreq'] != target_sfreq:
        raw = raw.copy().resample(target_sfreq, verbose=False)
    return raw


def bandpass_filter(raw):
    """Apply band-pass filter (0.5-50 Hz)."""
    return raw.copy().filter(
        l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
        method='fir', fir_design='firwin', verbose=False,
    )


def notch_filter(raw):
    """Apply notch filter to remove power line interference."""
    return raw.copy().notch_filter(
        freqs=NOTCH_FREQS, method='fir', fir_design='firwin', verbose=False,
    )


def run_ica(raw, n_components=20, random_state=42):
    """
    ICA artifact removal.
    Auto-detects EOG components using frontal channels, falls back to kurtosis.
    """
    ica = ICA(
        n_components=n_components, method='fastica',
        random_state=random_state, max_iter=500, verbose=False,
    )
    ica.fit(raw, verbose=False)

    # Auto-detect EOG components
    eog_indices = []
    frontal_chs = [ch for ch in ['Fp1', 'Fp2', 'F7', 'F8'] if ch in raw.ch_names]
    if frontal_chs:
        for ch in frontal_chs:
            try:
                indices, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
                eog_indices.extend(indices)
            except Exception:
                pass

    # Kurtosis fallback
    if not eog_indices:
        sources = ica.get_sources(raw).get_data()
        kurtosis = np.array([
            float(np.mean(s**4) / (np.mean(s**2)**2 + 1e-10) - 3)
            for s in sources
        ])
        threshold = np.mean(kurtosis) + 2 * np.std(kurtosis)
        eog_indices = list(np.where(kurtosis > threshold)[0])

    eog_indices = list(set(eog_indices))[:5]
    if eog_indices:
        ica.exclude = eog_indices

    return ica.apply(raw.copy(), verbose=False)


def harmonize_channels(raw, target_channels=None):
    """
    Map to canonical 32-channel template.
    Selects matching channels, drops extras, interpolates missing ones.
    """
    if target_channels is None:
        target_channels = COMMON_CHANNELS

    available = raw.ch_names
    ch_map = {}
    for target in target_channels:
        for avail in available:
            if avail.lower() == target.lower():
                ch_map[target] = avail
                break

    missing = [ch for ch in target_channels if ch not in ch_map]
    found_targets = [ch for ch in target_channels if ch in ch_map]
    found_names = [ch_map[ch] for ch in found_targets]

    if missing:
        print(f"[WARN] Missing channels: {missing} ", end="")

    # Pick found channels
    raw_picked = raw.copy().pick(found_names)

    # Rename to standardized names
    rename = {ch_map[t]: t for t in found_targets if ch_map[t] != t}
    if rename:
        raw_picked.rename_channels(rename)

    # Interpolate missing channels if any
    if missing and len(found_targets) >= 20:
        # Set montage for spatial interpolation
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw_picked.set_montage(montage, on_missing='ignore', verbose=False)

            # Create info for missing channels and add them as flat
            for ch_name in missing:
                info_ch = mne.create_info([ch_name], raw_picked.info['sfreq'], ch_types='eeg')
                flat_raw = mne.io.RawArray(
                    np.zeros((1, raw_picked.n_times)), info_ch, verbose=False
                )
                raw_picked = raw_picked.add_channels([flat_raw], force_update_info=True)

            raw_picked.set_montage(montage, on_missing='ignore', verbose=False)
            raw_picked.info['bads'] = missing
            raw_picked = raw_picked.interpolate_bads(verbose=False)
        except Exception as e:
            print(f"[interpolation failed: {e}] ", end="")

    # Reorder to canonical order
    available_final = raw_picked.ch_names
    ordered = [ch for ch in target_channels if ch in available_final]
    raw_picked = raw_picked.reorder_channels(ordered)

    return raw_picked


def segment_epochs(raw, duration=EPOCH_DURATION):
    """Segment continuous EEG into fixed-length non-overlapping epochs."""
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=duration - (1.0 / raw.info['sfreq']),
        baseline=None, preload=True, verbose=False,
    )
    return epochs


def preprocess_subject(subject, skip_ica=False):
    """Run full preprocessing pipeline on a single subject."""
    raw = subject.raw
    sid = subject.subject_id
    print(f"  Preprocessing {sid}... ", end="", flush=True)

    raw = resample(raw)
    raw = bandpass_filter(raw)
    raw = notch_filter(raw)

    if not skip_ica:
        try:
            raw = run_ica(raw)
        except Exception as e:
            print(f"[ICA skipped: {e}] ", end="")

    raw = harmonize_channels(raw)
    epochs = segment_epochs(raw)

    subject.epochs = epochs
    n_ep = len(epochs)
    n_ch = len(epochs.ch_names)
    n_t = epochs.get_data().shape[2]
    print(f"OK -> {n_ep} epochs x {n_ch} ch x {n_t} samples")
    return subject


def preprocess_all(subjects, skip_ica=False):
    """Run preprocessing on all subjects."""
    print(f"\n{'='*60}")
    print(f"PREPROCESSING")
    print(f"  Resample: {TARGET_SFREQ} Hz")
    print(f"  Bandpass: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print(f"  Notch: {NOTCH_FREQS} Hz")
    print(f"  ICA: {'Enabled' if not skip_ica else 'Disabled'}")
    print(f"  Channels: {len(COMMON_CHANNELS)}")
    print(f"  Epoch: {EPOCH_DURATION}s")
    print(f"{'='*60}")

    processed = []
    for subj in subjects:
        try:
            subj = preprocess_subject(subj, skip_ica=skip_ica)
            processed.append(subj)
        except Exception as e:
            print(f"  [ERROR] {subj.subject_id}: {e}")

    print(f"\nPreprocessed: {len(processed)}/{len(subjects)} subjects")
    return processed
