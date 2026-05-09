# Results Tables (paper-ready)

Both markdown and LaTeX forms. Numbers marked `[PENDING]` will be filled after
the 230-subject cross-dataset run completes. Preliminary numbers from the 30-subject
pilot run are shown in italic for internal reference — do not include in final paper.

---

## Table 1: Comparison with prior EEG-PD methods (main results table)

### Markdown version

| Method | Year | Architecture | Datasets | Protocol | Acc (%) | AUC | Notes |
|--------|------|-------------|----------|----------|---------|-----|-------|
| XGBoost (Anonymous) | 2025 | Shallow ML | Iowa only | Within-dataset | 79.0 | 0.86 | handcrafted features |
| LightCNN (2024) | 2024 | 1-layer CNN | Iowa | Within-dataset | 77.0 | 0.83 | minimal architecture |
| ASGCNN (Shah et al.) | 2023 | Sparse attention GCN | PD oddball | Within-dataset | 87.7 | — | single cohort |
| Multi-Head GSL (Nguyen et al.) | 2024 | Chebyshev GNN + explainer | UC San Diego | LOSO | 69.4 | — | single cohort |
| MCPNet (Qiu et al.) | 2024 | Multiscale CNN + prototypes | UNM, UC | Cross-dataset (2) | 90.2 / 86.5 | — | 2-dataset cross-test |
| ARP-N (Channel-Selected) | 2026 | CNN + channel harmonization | Iowa+UNM+UC | Stratified pooled CV | 80.6 | — | subjects pooled |
| TransformEEG (Cisotto et al.) | 2025 | Conv-Transformer | Iowa+UNM+UC+ds004148 | Nested-N-Subjects-Out | 80.1 | — | subjects pooled |
| **GNN-ProtoNet (GAT, ours)** | 2026 | GAT + ProtoNet + calibration | Iowa+UNM+UC | **Leave-one-dataset-out** | **[PENDING]** | **[PENDING]** | strictest cross-site |
| **GNN-ProtoNet (GCN, ours)** | 2026 | GCN + ProtoNet + calibration | Iowa+UNM+UC | **Leave-one-dataset-out** | **[PENDING]** | **[PENDING]** | ablation encoder |

*Internal preliminary (30-subject subset, skip-ICA, 20 epochs training, not final): GCN 73.6% / 0.816, GAT 67.8% / 0.766.*

### LaTeX version (drop into paper)

```latex
\begin{table}[t]
\centering
\caption{Comparison with prior EEG-based Parkinson's disease detection methods. Our GNN-ProtoNet is evaluated under a \emph{strict leave-one-dataset-out} protocol, in contrast to the pooled-subject protocols of recent cross-site baselines.}
\label{tab:main_results}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lllllcc}
\toprule
Method & Year & Architecture & Datasets & Protocol & Acc (\%) & AUC \\
\midrule
XGBoost baseline & 2025 & Shallow ML & Iowa & Within-dataset & 79.0 & 0.86 \\
LightCNN & 2024 & 1-layer CNN & Iowa & Within-dataset & 77.0 & 0.83 \\
ASGCNN \cite{shah2023asgcnn} & 2023 & Sparse attn. GCN & Oddball & Within-dataset & 87.7 & -- \\
Multi-Head GSL \cite{nguyen2024multihead} & 2024 & Chebyshev GNN & UC San Diego & LOSO & 69.4 & -- \\
MCPNet \cite{qiu2024mcpnet} & 2024 & Multiscale CNN + proto & UNM, UC & Cross-2 & 90.2 / 86.5 & -- \\
ARP-N \cite{arpn2026} & 2026 & CNN + harmonization & All 3 & Stratified pooled & 80.6 & -- \\
TransformEEG \cite{cisotto2025transformeeg} & 2025 & Conv-Transformer & All 3 + ds004148 & Nested-N-Subjects & 80.1 & -- \\
\midrule
\textbf{GNN-ProtoNet (GAT)} & 2026 & GAT + proto + calib. & All 3 & \textbf{LODO} & \textbf{[PENDING]} & \textbf{[PENDING]} \\
\textbf{GNN-ProtoNet (GCN)} & 2026 & GCN + proto + calib. & All 3 & \textbf{LODO} & \textbf{[PENDING]} & \textbf{[PENDING]} \\
\bottomrule
\end{tabular}%
}
\end{table}
```

---

## Table 2: Per-fold cross-dataset (LODO) results

| Train datasets | Test dataset | N_train | N_test | GAT Acc | GAT AUC | GCN Acc | GCN AUC |
|----------------|--------------|---------|--------|---------|---------|---------|---------|
| UC + UNM | Iowa | [PENDING] | 149 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| UC + Iowa | UNM | [PENDING] | 31 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| UNM + Iowa | UC | [PENDING] | 50 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| **Mean** | — | — | — | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

---

## Table 3: Ablation study (isolated contributions)

Reports accuracy under LODO on all 3 datasets, K=5. Delta is vs. full model.

| Configuration | Encoder | Calibration | K-shot | Acc (%) | Δ |
|---------------|---------|-------------|--------|---------|---|
| Full model | GAT | ✓ | 5 | [PENDING] | 0.0 |
| No calibration | GAT | ✗ | 5 | [PENDING] | −[PENDING] |
| GCN ablation | GCN | ✓ | 5 | [PENDING] | −[PENDING] |
| Random edges (no PLV) | GAT | ✓ | 5 | [PENDING] | −[PENDING] |
| K=1 | GAT | ✓ | 1 | [PENDING] | −[PENDING] |
| K=10 | GAT | ✓ | 10 | [PENDING] | +[PENDING] |
| K=20 | GAT | ✓ | 20 | [PENDING] | +[PENDING] |

---

## Table 4: Confusion matrix (per-fold)

For the best-performing fold (likely UNM+Iowa→UC based on pilot trends):

```
             Predicted PD   Predicted HC
Actual PD        TP              FN
Actual HC        FP              TN
```

| Fold | TP | FP | FN | TN | Sens | Spec | PPV | NPV |
|------|----|----|----|----|------|------|-----|-----|
| UC+UNM → Iowa | [P] | [P] | [P] | [P] | [P] | [P] | [P] | [P] |
| UC+Iowa → UNM | [P] | [P] | [P] | [P] | [P] | [P] | [P] | [P] |
| UNM+Iowa → UC | [P] | [P] | [P] | [P] | [P] | [P] | [P] | [P] |
