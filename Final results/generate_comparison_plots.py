"""
Generate publication-quality comparison plots for the GNN-ProtoNet paper.

Produces 4 figures in paper/figures/:
  1. comparison_bar.pdf     — Our method vs. prior SOTA (bar chart with error bars)
  2. lodo_per_fold.pdf      — Per-fold LODO accuracy and AUC
  3. kshot_curve.pdf        — Accuracy vs. K-shot (ablation)
  4. calibration_effect.pdf — With vs. without prototype calibration

Run after results are saved to results/cross_dataset_results.json.
If results are missing, plots render with placeholder bars so you can
still lay out the figure in the paper draft.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RESULTS_JSON = PROJECT / 'results' / 'cross_dataset_results.json'
FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def load_our_results():
    """Load subject-level results from the full-training and ensemble JSONs."""
    full_json = PROJECT / 'results' / 'cross_dataset_full_training.json'
    ens_json = PROJECT / 'results' / 'ensemble_results.json'

    out = {
        'gat_acc': 0.0, 'gat_auc': 0.0,
        'gcn_acc': 0.0, 'gcn_auc': 0.0,
        'ens_acc': 0.0, 'ens_auc': 0.0,
        'gat_folds': [], 'gcn_folds': [], 'ens_folds': [],
        'pilot': True,
    }

    if full_json.exists():
        with open(full_json) as f:
            d = json.load(f)
        # Use SUBJECT-LEVEL metrics (what prior work reports)
        out['gat_acc'] = d['gat'].get('mean_subject_level_accuracy',
                                       d['gat'].get('mean_accuracy', 0)) * 100
        out['gat_auc'] = d['gat'].get('mean_subject_level_auc',
                                       d['gat'].get('mean_auc', 0))
        out['gcn_acc'] = d['gcn'].get('mean_subject_level_accuracy',
                                       d['gcn'].get('mean_accuracy', 0)) * 100
        out['gcn_auc'] = d['gcn'].get('mean_subject_level_auc',
                                       d['gcn'].get('mean_auc', 0))
        # Per-fold: pull subject-level if present
        for f in d['gat'].get('folds', []):
            out['gat_folds'].append({
                'train_datasets': f['train_datasets'],
                'test_dataset': f['test_dataset'],
                'accuracy': f.get('subject_level_accuracy', f.get('accuracy', 0)),
                'auc_roc': f.get('subject_level_auc', f.get('auc_roc', 0)),
            })
        for f in d['gcn'].get('folds', []):
            out['gcn_folds'].append({
                'train_datasets': f['train_datasets'],
                'test_dataset': f['test_dataset'],
                'accuracy': f.get('subject_level_accuracy', f.get('accuracy', 0)),
                'auc_roc': f.get('subject_level_auc', f.get('auc_roc', 0)),
            })
        out['pilot'] = False

    if ens_json.exists():
        with open(ens_json) as f:
            d = json.load(f)
        out['ens_acc'] = d.get('mean_subject_accuracy', 0) * 100
        out['ens_auc'] = d.get('mean_subject_auc', 0)
        for f in d.get('folds', []):
            out['ens_folds'].append({
                'train_datasets': f['train_datasets'],
                'test_dataset': f['test_dataset'],
                'accuracy': f.get('subject_acc', 0),
                'auc_roc': f.get('subject_auc', 0),
            })
    return out


def fig1_comparison_bar(ours):
    """Bar chart: our method vs. prior SOTA on same 3 datasets.
    Horizontal layout for readability (long method names)."""
    # Order: oldest/lowest first, ours last for emphasis
    methods = [
        'Neves et al. (2024)',
        'TransformEEG (Del Pup et al., 2025)',
        'ARP-N (2026)',
        'Chang et al. ASGCNN (2023)',
        'MCPNet (Qiu et al., 2024)*',
        'GNN-ProtoNet GAT (ours)',
        'GNN-ProtoNet GCN (ours)',
        'GNN-ProtoNet Ensemble (ours)',
    ]
    acc = [69.4, 80.1, 80.6, 87.7, 90.2,
           ours['gat_acc'], ours['gcn_acc'], ours['ens_acc']]
    protocols = ['Subject LOSO (1 site)',
                 'Pooled N-Subjects-Out',
                 'Pooled Stratified CV',
                 'Within-dataset',
                 'Cross-dataset (2 sites)',
                 'Strict LODO (3 sites)',
                 'Strict LODO (3 sites)',
                 'Strict LODO (3 sites)']
    # Gray for prior work, ours in blue/green/red
    colors = ['#BDBDBD'] * 5 + ['#1565C0', '#2E7D32', '#C62828']

    # Horizontal bars — much more readable for long method names
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(methods))
    bars = ax.barh(y, acc, color=colors, edgecolor='black', linewidth=0.7,
                   height=0.65)

    # Value labels at the right end of each bar
    for bar, val in zip(bars, acc):
        ax.text(val + 0.6, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', ha='left', va='center', fontsize=10,
                fontweight='bold')

    # Y-axis: method name + protocol on second line
    ylabels = [f'{m}\n({p})' for m, p in zip(methods, protocols)]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()  # best at top

    ax.set_xlabel('Subject-level accuracy (%)', fontsize=11)
    ax.set_xlim(60, 105)
    ax.set_xticks([60, 70, 80, 90, 100])

    # Reference line at the strongest non-LODO baseline (MCPNet 90.2%)
    ax.axvline(90.2, color='#777', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.text(90.2, -0.7, 'MCPNet\n(2-dataset)',
            ha='center', va='bottom', fontsize=8, color='#555', style='italic')

    ax.set_title("Cross-site Parkinson's EEG detection — method comparison",
                 fontsize=12, pad=14)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Footnote below the plot, with proper spacing
    fig.text(0.02, 0.005,
             '*MCPNet reports on 2 datasets only (UNM+UC). All other rows use 3 OpenNeuro datasets. '
             'Protocols differ in stringency; LODO is the strictest cross-site test.',
             fontsize=7.5, style='italic', color='#555')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIG_DIR / 'comparison_bar.pdf')
    plt.savefig(FIG_DIR / 'comparison_bar.png')
    plt.close()
    print(f'  Wrote {FIG_DIR}/comparison_bar.pdf')


def fig2_lodo_per_fold(ours):
    """Per-fold LODO subject-level accuracy and AUC for GAT, GCN, Ensemble."""
    if ours['gat_folds']:
        folds = [f"{'+'.join(f['train_datasets'])}\n→ {f['test_dataset']}"
                 for f in ours['gat_folds']]
        gat_acc = [f.get('accuracy', 0) * 100 for f in ours['gat_folds']]
        gcn_acc = [f.get('accuracy', 0) * 100 for f in ours['gcn_folds']]
        ens_acc = [f.get('accuracy', 0) * 100 for f in ours['ens_folds']] \
            if ours.get('ens_folds') else [0] * len(folds)
        gat_auc = [f.get('auc_roc', 0) for f in ours['gat_folds']]
        gcn_auc = [f.get('auc_roc', 0) for f in ours['gcn_folds']]
        ens_auc = [f.get('auc_roc', 0) for f in ours['ens_folds']] \
            if ours.get('ens_folds') else [0] * len(folds)
    else:
        folds = ['UNM+Iowa → UC', 'UC+Iowa → UNM', 'UC+UNM → Iowa']
        gat_acc = gcn_acc = ens_acc = [0, 0, 0]
        gat_auc = gcn_auc = ens_auc = [0, 0, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(folds))
    w = 0.26

    # Left: accuracy
    b1 = ax1.bar(x - w, gat_acc, w, label='GAT', color='#1565C0',
                 edgecolor='black', linewidth=0.6)
    b2 = ax1.bar(x,     gcn_acc, w, label='GCN', color='#2E7D32',
                 edgecolor='black', linewidth=0.6)
    b3 = ax1.bar(x + w, ens_acc, w, label='Ensemble', color='#C62828',
                 edgecolor='black', linewidth=0.6)
    for bars in (b1, b2, b3):
        for bar in bars:
            v = bar.get_height()
            if v > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, v + 0.6,
                         f'{v:.1f}', ha='center', va='bottom', fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds, fontsize=9.5)
    ax1.set_ylabel('Subject-level accuracy (%)', fontsize=11)
    ax1.set_ylim(60, 105)
    # Reference lines at top of plot, not crashing into bars
    ax1.axhline(80.10, color='#FF9800', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axhline(80.60, color='#F44336', linestyle=':', linewidth=1.2, alpha=0.7)
    ax1.text(len(folds) - 0.5, 80.10, ' TransformEEG (80.1%)',
             color='#E65100', fontsize=8, va='center', ha='right',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                       edgecolor='none', alpha=0.85))
    ax1.text(len(folds) - 0.5, 78.0, ' ARP-N (80.6%)',
             color='#B71C1C', fontsize=8, va='center', ha='right',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                       edgecolor='none', alpha=0.85))
    ax1.set_title('(a) Accuracy', fontsize=11)
    ax1.legend(fontsize=9, loc='lower left', framealpha=0.95,
               edgecolor='#888')
    ax1.grid(axis='y', alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: AUC
    b1 = ax2.bar(x - w, gat_auc, w, label='GAT', color='#1565C0',
                 edgecolor='black', linewidth=0.6)
    b2 = ax2.bar(x,     gcn_auc, w, label='GCN', color='#2E7D32',
                 edgecolor='black', linewidth=0.6)
    b3 = ax2.bar(x + w, ens_auc, w, label='Ensemble', color='#C62828',
                 edgecolor='black', linewidth=0.6)
    for bars in (b1, b2, b3):
        for bar in bars:
            v = bar.get_height()
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                         f'{v:.3f}', ha='center', va='bottom', fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(folds, fontsize=9.5)
    ax2.set_ylabel('Subject-level AUC-ROC', fontsize=11)
    ax2.set_ylim(0.80, 1.03)
    ax2.set_title('(b) AUC-ROC', fontsize=11)
    ax2.legend(fontsize=9, loc='lower left', framealpha=0.95,
               edgecolor='#888')
    ax2.grid(axis='y', alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'lodo_per_fold.pdf')
    plt.savefig(FIG_DIR / 'lodo_per_fold.png')
    plt.close()
    print(f'  Wrote {FIG_DIR}/lodo_per_fold.pdf')


def fig3_kshot_curve():
    """K-shot ablation curve. Skipped if no ablation data exists yet."""
    ablation_path = PROJECT / 'results' / 'kshot_ablation.json'
    if not ablation_path.exists():
        print(f'  Skip kshot_curve: run K-shot ablation first to generate {ablation_path}')
        return

    with open(ablation_path) as f:
        d = json.load(f)
    k_values = d['k_values']
    gat_acc = d['gat_acc']
    gcn_acc = d['gcn_acc']

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(k_values, gat_acc, 'o-', color='#1565C0', label='GAT',
            linewidth=2, markersize=8)
    ax.plot(k_values, gcn_acc, 's-', color='#2E7D32', label='GCN',
            linewidth=2, markersize=8)
    ax.set_xlabel('K-shot (support samples per class)', fontsize=11)
    ax.set_ylabel('Subject-level accuracy (%)', fontsize=11)
    ax.set_xticks(k_values)
    ax.set_ylim(60, 100)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    ax.set_title('K-shot ablation under strict LODO', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'kshot_curve.pdf')
    plt.savefig(FIG_DIR / 'kshot_curve.png')
    plt.close()
    print(f'  Wrote {FIG_DIR}/kshot_curve.pdf')


def fig4_calibration_effect():
    """Calibration on vs. off — skip if no ablation data."""
    ablation_path = PROJECT / 'results' / 'calibration_ablation.json'
    if not ablation_path.exists():
        print(f'  Skip calibration_effect: run calibration ablation first')
        return
    # When data is available, render the proper figure here.
    print(f'  TODO: implement calibration_effect plot once ablation JSON is created.')


def fig5_confusion_matrices(ours):
    """Subject-level confusion matrices for each LODO fold (ensemble)."""
    if not ours.get('ens_folds'):
        print('  Skip confusion matrices: no ensemble folds')
        return
    ens_path = PROJECT / 'results' / 'ensemble_results.json'
    if not ens_path.exists():
        print('  Skip confusion matrices: ensemble JSON missing')
        return
    with open(ens_path) as f:
        d = json.load(f)
    folds = d.get('folds', [])
    if not folds:
        return

    fig, axes = plt.subplots(1, len(folds), figsize=(3.6 * len(folds), 3.4))
    if len(folds) == 1:
        axes = [axes]
    for ax, f in zip(axes, folds):
        cm = np.array([[f.get('tn', 0), f.get('fp', 0)],
                       [f.get('fn', 0), f.get('tp', 0)]])
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                color = 'white' if v > cm.max() * 0.5 else '#222'
                ax.text(j, i, str(v), ha='center', va='center',
                        fontsize=14, fontweight='bold', color=color)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred HC', 'Pred PD'], fontsize=9)
        ax.set_yticklabels(['True HC', 'True PD'], fontsize=9)
        title = (f"{'+'.join(f['train_datasets'])} → {f['test_dataset']}\n"
                 f"Acc {f.get('subject_acc', 0)*100:.2f}%  "
                 f"AUC {f.get('subject_auc', 0):.3f}")
        ax.set_title(title, fontsize=10)
    plt.suptitle('Subject-level confusion matrices (Ensemble, LODO)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'confusion_matrices.pdf')
    plt.savefig(FIG_DIR / 'confusion_matrices.png')
    plt.close()
    print(f'  Wrote {FIG_DIR}/confusion_matrices.pdf')


def main():
    print('Loading results...')
    ours = load_our_results()
    if ours['pilot']:
        print('  WARNING: Using pilot 30-subject numbers (full run not complete).')
    else:
        print(f'  Full results loaded: GAT {ours["gat_acc"]:.1f}% / '
              f'GCN {ours["gcn_acc"]:.1f}%')

    print('\nGenerating plots...')
    fig1_comparison_bar(ours)
    fig2_lodo_per_fold(ours)
    fig3_kshot_curve()
    fig4_calibration_effect()
    fig5_confusion_matrices(ours)

    print(f'\nAll plots in: {FIG_DIR}')


if __name__ == '__main__':
    main()
