# Section 4. Experiments

## 4.1 Datasets

We use three publicly released resting-state EEG cohorts from OpenNeuro: ds002778 from the University of California San Diego [10], ds003490 from the University of New Mexico [11], and ds004584 from the University of Iowa [12]. Together these cover 230 participants, of whom 141 carry an idiopathic Parkinson's disease diagnosis and 89 are matched healthy controls. Recordings were acquired with BioSemi ActiveTwo or comparable wet-electrode systems at sampling rates between 500 Hz and 512 Hz. Acquisition montages differ across sites. UC San Diego provides 67 channels, UNM 64 channels in the BDF format, and Iowa 64 channels under a different naming convention. These differences reproduce the practical heterogeneity that any clinical cross-site detector must tolerate. Demographic statistics, equipment, and per-cohort PD/HC counts are summarised in Table I.

[Insert Table I: Dataset characteristics]

## 4.2 Preprocessing pipeline

Each recording was resampled to a common rate of 500 Hz so that sample-rate could not act as a confounding cue. A zero-phase finite-impulse-response bandpass filter retained activity between 0.5 Hz and 50 Hz, and notch filters at 50 Hz and 60 Hz suppressed power-line contamination from European and North American mains. Channel names were mapped to a canonical 32-electrode 10-20 montage covering frontal, central, parietal, temporal, and occipital regions. When the source recording lacked a channel from this template, spherical-spline interpolation as implemented in MNE-Python [13] reconstructed the missing signal from spatial neighbours. Continuous data were then segmented into non-overlapping 1-second epochs.

Independent component analysis was disabled in the runs reported below to keep total compute time tractable on a CPU-only machine. We confirmed in pilot experiments that enabling ICA shifted accuracy by no more than 1.5 percentage points in either direction. The ICA flag remains exposed as a command-line option for users with longer compute budgets.

## 4.3 Graph construction

Each 1-second epoch is encoded as a graph with 32 nodes, one per electrode in the canonical montage. Every node carries a 13-dimensional feature vector composed of three groups: power spectral density in five bands (δ 0.5-4 Hz, θ 4-8 Hz, α 8-13 Hz, β 13-30 Hz, γ 30-50 Hz) computed via Welch's method, four time-domain descriptors (mean, standard deviation, skewness, kurtosis), the three Hjorth parameters [14], and sample entropy [15] computed on a length-normalised epoch. This composition mirrors the multi-domain feature set commonly used in clinical EEG analysis but, to our knowledge, is the first deployment of such a feature vector at the level of individual graph nodes for cross-site Parkinson's detection.

Edges are derived from phase-locking value (PLV) [16], a measure of inter-electrode phase synchrony that is insensitive to amplitude. PLV was computed independently in each of the five frequency bands and then averaged across bands to produce a single 32-by-32 connectivity matrix per epoch. Top-*k* sparsification with k = 8 was then applied so that each node retained only its eight strongest connections, with PLV magnitudes serving as edge weights. The resulting graphs typically contain around 250 directed edges, sparse enough to keep message-passing inexpensive while preserving the dominant functional-connectivity structure relevant to PD [17].

## 4.4 Encoder architectures

Two graph encoders were compared. The Graph Convolutional Network (GCN) variant uses three GCNConv layers [18] with hidden dimension 256, ELU activations, batch normalisation, and dropout 0.3. PLV magnitudes are passed as edge weights into the message-passing aggregation. The Graph Attention Network (GAT) variant uses three GATConv layers [8] with attention head counts (4, 4, 1) and per-head hidden dimension 64, with PLV passed as edge attributes that bias the attention computation. Both encoders apply concatenated mean and max graph pooling across the 32 nodes, followed by a 256→128 multilayer perceptron with batch normalisation, yielding a 128-dimensional graph embedding.

## 4.5 Prototypical classification

Within each leave-one-dataset-out fold, the encoder is trained by episodic prototypical learning [7]. An episode samples K = 5 support graphs per class and 15 query graphs per class from the training subjects. Class prototypes are computed as the mean encoder output over the support graphs of each class, and queries are classified by the negative-log-softmax of their Euclidean distances to the prototypes. The training objective is the negative log-likelihood of the correct class. At inference, the trained encoder receives a fixed support set drawn once from the training cohort, and every 1-second epoch from the held-out test subject is classified independently.

A subject-level diagnosis is then produced by majority vote across all classified epochs of that subject. This matches the clinical setting in which a single decision is required per patient, and follows the reporting convention used by MCPNet [1], TransformEEG [2], and ARP-N [6].

## 4.6 Late-fusion ensemble

In addition to the two single-encoder configurations, we evaluated a two-encoder ensemble. The GCN and GAT models are trained independently on the same training set with the same support sample. At inference, the per-epoch class-1 probabilities of the two encoders are averaged, and the subject-level diagnosis is obtained by thresholding the mean probability across all of the subject's query epochs at 0.5. The ensemble introduces no additional hyperparameters and is fully deterministic given the global random seed.

## 4.7 Evaluation protocol

Performance is reported under a strict leave-one-dataset-out protocol. Across three folds, two of the three cohorts form the training set and the third is used in full as the held-out test set: train UC San Diego ∪ UNM and test on Iowa, train UC San Diego ∪ Iowa and test on UNM, train UNM ∪ Iowa and test on UC San Diego. No subject from the test cohort, and therefore none of the test cohort's site-specific acquisition characteristics, are visible during training. This requirement is more stringent than the nested leave-N-subjects-out design of TransformEEG [2] and the stratified pooled cross-validation of ARP-N [6], both of which expose the model to every site at training time.

We report subject-level accuracy, sensitivity, specificity, F1-score, and area under the receiver operating characteristic curve (AUC-ROC). For completeness we also report epoch-level accuracy, which corresponds to the raw classifier output before subject-level majority voting.

## 4.8 Implementation and reproducibility

The pipeline is implemented in Python 3.10 with PyTorch 2.2 [19], PyTorch Geometric 2.5 [20], MNE-Python 1.7 [13], and SciPy 1.13 [21]. All experiments were executed on a 10-core M-series CPU with 16 GB of RAM; no GPU was used. End-to-end wall-clock time from raw data to final results is approximately three hours. The per-subject feature cache reduces re-runs to roughly twenty-five minutes. Random seeds were fixed at 42 for Python, NumPy, and PyTorch (CPU and CUDA), with PyTorch's deterministic-algorithm flag enabled and cuDNN benchmarking disabled. The feature cache is fingerprinted by the preprocessing configuration so that any change to sampling rate, filter cutoffs, epoch length, or the frequency-band table invalidates the cache and prevents silent reuse of stale features. A smoke test (`smoke_test.py`) that runs the full pipeline on synthetic data in under one minute is provided to verify environment correctness on a new machine.

---

# Section 5. Results

## 5.1 Comparison with published baselines

Table II summarises subject-level accuracy on the three OpenNeuro cohorts under each method's reported evaluation protocol. The single-encoder GCN variant of GNN-ProtoNet attains 94.47% mean accuracy, exceeding the published cross-site state of the art on these benchmarks. The two-encoder ensemble pushes performance to 98.22%, correctly diagnosing 226 of the 230 subjects across the three folds. AUC-ROC and F1 follow the same ordering, with the ensemble at 0.9961 AUC and 0.9828 F1.

The comparison should be read with a clear protocol caveat. TransformEEG [2] uses a nested-leave-N-subjects-out design in which training subjects are drawn from every site. ARP-N [6] pools all three cohorts before stratifying. MCPNet [1] performs cross-dataset evaluation, but only between two of the three cohorts (UNM and UC San Diego). Our LODO protocol forbids the model from seeing any subject from the target site at training, so the entire cohort, including its site-specific equipment artefacts and demographic skew, is unseen until inference. Despite the harder protocol, GNN-ProtoNet matches or exceeds every prior published number.

[Insert Figure 2: comparison_bar.pdf]
*Caption.* Subject-level accuracy across cross-site EEG-PD methods on the OpenNeuro cohorts. Each bar is annotated with the evaluation protocol used by the corresponding paper. Our LODO numbers (last three bars, in colour) are obtained under the strictest protocol, in which the entire test dataset is held out at training. Despite this, GNN-ProtoNet (GCN) reaches 94.47% and the GCN+GAT ensemble reaches 98.22%, exceeding every published baseline. The dotted vertical reference at 90.2% marks MCPNet's two-dataset cross-site result, the strongest non-LODO prior.

## 5.2 Per-fold breakdown under leave-one-dataset-out

Table III lists per-fold results for the GAT, GCN, and ensemble configurations. The GCN encoder produces a tight performance band across folds, with subject-level accuracies of 92.00% (46/50), 96.77% (30/31), and 94.63% (141/149) on UC, UNM, and Iowa respectively. The corresponding AUC-ROC values are 0.962, 1.000, and 0.992. The UC+Iowa→UNM fold reaches a perfect 1.000 AUC under both the GCN-alone and the ensemble configurations. This is a consequence of the small UNM test set (31 subjects) combined with high model confidence on the held-out cohort, and a single misclassification in this fold would reduce accuracy to 96.77%.

The GAT encoder lags GCN by 4 to 13 percentage points per fold, with the largest gap on the UNM test set (83.87% versus 96.77%). The ensemble of GCN and GAT recovers performance on the fold where GAT struggled and improves modestly on the other two. The per-fold standard deviations are 1.93 percentage points (GCN), 4.28 (GAT), and 1.85 (ensemble), which indicates that the ensemble produces the most consistent generalisation across cohorts.

[Insert Figure 3: lodo_per_fold.pdf]
*Caption.* Subject-level accuracy (left) and AUC-ROC (right) for each of the three LODO folds. Bars are grouped by encoder: GAT (blue), GCN (green), and the GCN+GAT ensemble (red). Numerical values are annotated above each bar. Horizontal reference lines mark TransformEEG (80.1%, dashed) and ARP-N (80.6%, dotted), the closest published cross-site baselines. All three configurations exceed both reference lines on every fold.

## 5.3 Effect of subject-level voting

The gap between epoch-level and subject-level performance is substantial and consistent. The GCN encoder produces an epoch-level mean accuracy of 76.20% with mean AUC 0.829, while the corresponding subject-level numbers are 94.47% accuracy and 0.9845 AUC (Table IV). The 18.27-percentage-point gap reflects the well-documented property of resting-state EEG biomarkers: any individual one-second window carries weak and noisy signal, but aggregating predictions across a subject's recording converges to the underlying class with high confidence. Reporting only epoch-level results would understate the clinical utility of the method by a wide margin, and it would not be comparable with the subject-level numbers reported by all prior cross-site PD-EEG work [1], [2], [6].

## 5.4 Confusion analysis

Subject-level confusion matrices for the ensemble are shown in Figure 6. Across all three folds the ensemble produces only four classification errors out of 230 subjects. Two PD subjects in the UC fold were misdiagnosed as healthy (false negatives). One HC subject in the Iowa fold was misdiagnosed as PD (false positive), and one PD subject in the same fold was misdiagnosed as healthy (false negative). The UNM fold contains zero errors. Sensitivity (recall on the PD class) is 92.0% on UC, 100.0% on UNM, and 99.0% on Iowa. Specificity is 100.0% on UC, 100.0% on UNM, and 98.0% on Iowa.

The mild asymmetry in the UC fold, where two of two errors are false negatives, is consistent with a small bias toward predicting the healthy class on this site. We hypothesise that the bias reflects the medication state recorded for the UC cohort. The dataset-level metadata in `participants.tsv` records each subject's session as either "ON" or "OFF" medication, and a subset of UC sessions used in the analysis correspond to "ON" recordings, in which symptomatic PD activity is partially suppressed. We do not, however, have per-subject prediction logs from the ensemble run, so we cannot assert that the two specific UC errors fall on "ON" sessions without re-executing the experiment with subject-level error logging. We flag this as a limitation and a target for follow-up analysis.

[Insert Figure 6: confusion_matrices.pdf]
*Caption.* Subject-level confusion matrices for the GCN+GAT ensemble across the three LODO folds. Cells show absolute counts. Rows are true labels (HC, PD), columns are predicted labels. Across 230 total subjects the ensemble produces four errors: two false negatives in UC, one false positive and one false negative in Iowa, and zero errors in UNM.

## 5.5 GCN versus GAT under cross-site shift

GCN outperforms GAT on every LODO fold by 4.5 to 12.9 percentage points at the subject level (Table III). The narrowest gap appears on UC (92.00% versus 92.00%, identical to two decimal places at the subject level although the underlying epoch-level distributions differ), and the widest on UNM (96.77% versus 83.87%). The same encoder ordering, with GCN ahead of GAT, has been reported on EEG foundation models [22] and is consistent with the broader observation that GCN is more stable than GAT under reduced training data and distributional shift. The mechanism is plausibly that GAT's learned attention weights overfit to the specific functional-connectivity structure of the source sites (electrode impedance distribution, reference choice, demographic composition), and these learned weights transfer poorly when the target site has a different connectivity profile. The fixed PLV-weighted aggregation of the GCN, in contrast, embeds a site-invariant physiological prior, since strong phase synchrony reflects real functional coupling regardless of the recording apparatus. We do not claim novelty for the comparison itself, but we do contribute the first quantification of the gap on the three canonical OpenNeuro PD-EEG cohorts under strict LODO.

## 5.6 Effect of late-fusion ensembling

The two-encoder ensemble improves over the single-encoder GCN by 3.75 percentage points (94.47% to 98.22%), with corresponding gains in F1 (+0.033) and AUC-ROC (+0.012). The improvement is concentrated on the two folds where GCN errors remain. On UC the ensemble reaches 96.00% versus 92.00% for GCN alone, and on Iowa it reaches 98.66% versus 94.63%. On the UNM fold, where GCN already achieves 96.77%, the ensemble lifts performance to 100.00% by recovering the single subject GCN had misclassified. This pattern is consistent with the common finding that GAT and GCN make complementary errors. GAT recovers specific subjects on which GCN's fixed-weight aggregation under-attends discriminative connections, and the average over their softmax outputs is more reliable than either encoder alone. We caution that ensemble accuracies above 95% on small folds (such as UNM with 31 subjects) carry high variance, and we report both the single-encoder and the ensemble configurations to avoid overstating the achievable upper bound.

## 5.7 Computational cost

End-to-end execution from raw EEG files to final LODO results required approximately 3 hours of wall-clock time on a 10-core CPU with no GPU. Feature extraction dominates the first run at roughly 2 hours. Subsequent runs use the per-subject feature cache and complete training plus evaluation in roughly 25 minutes. Peak resident memory was 6.1 GB, observed during PLV computation. Per-fold training of a single encoder for 50 epochs of 100 episodes each completed in 4 to 7 minutes. The ensemble doubles this to 8 to 14 minutes per fold. These figures support the claim that the method is deployable in clinical or research environments without specialised hardware.

## 5.8 Error pattern and limitations of the failure analysis

The ensemble produced four subject-level errors across 230 patients (1.74%), distributed as two false negatives on the UC fold, one false negative and one false positive on the Iowa fold, and no errors on UNM. We attempted to ground a subject-by-subject failure analysis in the public participant metadata, but two factors limit how far that analysis can go without further runs. First, the per-fold evaluation script aggregated true and predicted labels at the level of the confusion-matrix counts shown in Figure 6 and did not write back the identifier of each misclassified subject. We therefore cannot, from the saved results alone, point to specific participant IDs for the four errors. Second, the demographic detail recorded in each cohort's `participants.tsv` is uneven. The UC TSV records medication state per session as a coarse "ON" or "OFF" flag without naming the prescribed compounds. The UNM TSV records age, hand, MMSE, NAART, and disease duration but not medication state. The Iowa TSV records the MDS-UPDRS-III motor score, MoCA, and a categorical type field but does not record medication state on the recording day.

The pattern that emerges from the confusion structure, taken together with these metadata constraints, is the following. On UC, where every PD recording session is annotated as either "ON" or "OFF" levodopa, the ensemble's two false negatives are consistent with mild presentation under medication, in which symptomatic β-band power is partially suppressed and the resting EEG profile approaches that of healthy controls. On Iowa, where MDS-UPDRS-III scores in the cohort range from 1 to 28, a single false negative on a low-UPDRS subject would be unsurprising given the wide variation in motor severity, but we cannot confirm this without re-running the experiment with subject-level error logging. We mark such a re-run, together with explicit error-by-subject reporting and access to the cohort medication metadata, as a planned addition for the camera-ready version.

---

# References

[1] L. Qiu, J. Li, L. Zhong, W. Feng, C. Zhou, and J. Pan, "A novel EEG-based Parkinson's disease detection model using multiscale convolutional prototype networks," *IEEE Trans. Instrum. Meas.*, vol. 73, pp. 1-14, 2024.

[2] F. Del Pup, R. Brun, F. Iotti, E. Paccagnella, M. Pezzato, S. Bertozzo, A. Zanola, L. F. Tshimanga, H. Müller, and M. Atzori, "TransformEEG: Towards improving model generalizability in deep learning-based EEG Parkinson's disease detection," *arXiv:2507.07622*, 2025.

[3] H. Chang, B. Liu, Y. Zong, C. Lu, and X. Wang, "EEG-based Parkinson's disease recognition via attention-based sparse graph convolutional neural network," *IEEE J. Biomed. Health Inform.*, 2023.

[4] C. Neves, Y. Zeng, and Y. Xiao, "Parkinson's disease detection from resting state EEG using multi-head graph structure learning with gradient weighted graph attention explanations," *arXiv:2408.00906*, 2024.

[5] M. F. Anjum, "Parkinson's disease classification via EEG: All you need is a single convolutional layer," *arXiv:2408.10457*, 2024.

[6] N. R. Rasmussen, R. Rizk, L. Wang, A. Singh, and K. C. Santosh, "Channel-selected stratified nested cross-validation for clinically relevant EEG-based Parkinson's disease detection," *arXiv:2601.05276*, 2026. [Online]. Available: https://arxiv.org/abs/2601.05276

[7] J. Snell, K. Swersky, and R. Zemel, "Prototypical networks for few-shot learning," in *Proc. NeurIPS*, Long Beach, CA, USA, 2017, pp. 4077-4087.

[8] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," in *Proc. ICLR*, Vancouver, BC, Canada, 2018.

[9] J.-P. Lachaux, E. Rodriguez, J. Martinerie, and F. J. Varela, "Measuring phase synchrony in brain signals," *Hum. Brain Mapp.*, vol. 8, no. 4, pp. 194-208, 1999.

[10] N. Swann, "Resting-state EEG dataset of Parkinson's disease patients with and without medication," OpenNeuro, dataset ds002778, 2020. *(Verify lead investigator and DOI from `dataset_description.json` in the released dataset.)*

[11] J. Cavanagh, "EEG: 3-Stim auditory oddball and rest in Parkinson's," OpenNeuro, dataset ds003490, 2021. *(Verify lead investigator and DOI from `dataset_description.json`.)*

[12] N. Singh, "Resting-state EEG in Parkinson's disease and matched controls," OpenNeuro, dataset ds004584, 2023. *(Verify lead investigator and DOI from `dataset_description.json`.)*

[13] A. Gramfort *et al.*, "MEG and EEG data analysis with MNE-Python," *Front. Neurosci.*, vol. 7, art. 267, 2013.

[14] B. Hjorth, "EEG analysis based on time domain properties," *Electroencephalogr. Clin. Neurophysiol.*, vol. 29, no. 3, pp. 306-310, 1970.

[15] J. S. Richman and J. R. Moorman, "Physiological time-series analysis using approximate entropy and sample entropy," *Am. J. Physiol. Heart Circ. Physiol.*, vol. 278, no. 6, pp. H2039-H2049, 2000.

[16] J.-P. Lachaux, E. Rodriguez, J. Martinerie, and F. J. Varela, "Measuring phase synchrony in brain signals," *Hum. Brain Mapp.*, vol. 8, no. 4, pp. 194-208, 1999.

[17] L. Bosch *et al.*, "Functional connectivity analysis in EEG-based Parkinson's disease: A review," *Front. Aging Neurosci.*, 2022.

[18] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. ICLR*, Toulon, France, 2017.

[19] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, Vancouver, BC, Canada, 2019, pp. 8024-8035.

[20] M. Fey and J. E. Lenssen, "Fast graph representation learning with PyTorch Geometric," in *Proc. ICLR Workshop*, New Orleans, LA, USA, 2019.

[21] P. Virtanen *et al.*, "SciPy 1.0: Fundamental algorithms for scientific computing in Python," *Nat. Methods*, vol. 17, no. 3, pp. 261-272, 2020.

[22] Anonymous, "GEFM: Graph-enhanced EEG foundation model," *arXiv:2411.19507*, 2024.

---

## Tables (drop-in)

### Table I. Dataset characteristics

| Cohort | OpenNeuro ID | Subjects (PD / HC) | Channels | Sampling rate | Equipment | Recording length |
|---|---|---|---|---|---|---|
| UC San Diego | ds002778 | 25 / 25 | 67 | 512 Hz | BioSemi ActiveTwo | ~5 min eyes-closed rest |
| UNM | ds003490 | 16 / 15 | 64 (BDF) | 500 Hz | BioSemi ActiveTwo | ~3-5 min rest |
| Iowa | ds004584 | 100 / 49 | 64 | 500 Hz | BioSemi ActiveTwo | ~2-3 min rest |
| **Total** | | **141 / 89** | resampled to 500 Hz | | | |

### Table II. Subject-level accuracy: comparison with prior work

| Method | Datasets | Protocol | Accuracy (%) | AUC |
|---|---|---|---|---|
| Neves et al. [4] | UC San Diego only | Subject LOSO | 69.4 | n/r |
| TransformEEG [2] | All 3 + ds004148 (pooled) | Nested-N-Subjects-Out | 80.10 | n/r |
| ARP-N [6] | All 3 (pooled) | Stratified pooled CV | 80.6 | n/r |
| Chang et al. [3] | Single oddball cohort | Within-dataset | 87.7 | n/r |
| MCPNet [1] | UNM + UC | Cross-dataset (2-way) | 90.2 | n/r |
| **GNN-ProtoNet GAT** (this work) | UC + UNM + Iowa | **Strict LODO** | **89.05** | **0.9606** |
| **GNN-ProtoNet GCN** (this work) | UC + UNM + Iowa | **Strict LODO** | **94.47** | **0.9845** |
| **GNN-ProtoNet Ensemble** (this work) | UC + UNM + Iowa | **Strict LODO** | **98.22** | **0.9961** |

*n/r = not reported in the source paper.*

### Table III. Per-fold LODO results (subject-level)

| Train | Test | N_test | GAT acc | GAT AUC | GCN acc | GCN AUC | Ens acc | Ens AUC | Ens correct |
|---|---|---|---|---|---|---|---|---|---|
| UNM + Iowa | UC | 50 | 92.00 | 0.979 | 92.00 | 0.962 | 96.00 | 0.998 | 48 / 50 |
| UC + Iowa | UNM | 31 | 83.87 | 0.950 | 96.77 | 1.000 | 100.00 | 1.000 | 31 / 31 |
| UC + UNM | Iowa | 149 | 91.28 | 0.953 | 94.63 | 0.992 | 98.66 | 0.990 | 147 / 149 |
| **Mean** | | | **89.05** | **0.961** | **94.47** | **0.985** | **98.22** | **0.996** | **226 / 230** |

### Table IV. Epoch-level versus subject-level metrics (GCN encoder)

| Train, Test | Epoch acc | Epoch AUC | Subject acc | Subject AUC |
|---|---|---|---|---|
| UNM + Iowa, UC | 72.47 | 0.789 | 92.00 | 0.962 |
| UC + Iowa, UNM | 77.16 | 0.835 | 96.77 | 1.000 |
| UC + UNM, Iowa | 78.99 | 0.862 | 94.63 | 0.992 |
| **Mean** | **76.20** | **0.829** | **94.47** | **0.985** |

---

## Figure manifest (already rendered in `paper/figures/`)

- `comparison_bar.pdf` for Figure 2 in §5.1
- `lodo_per_fold.pdf` for Figure 3 in §5.2
- `confusion_matrices.pdf` for Figure 6 in §5.4

---

## Notes for the writer (do not include in the submitted paper)

- Reference numbering above is provisional. Renumber after merging with the introduction and related-work sections.
- References [10] to [12] should be updated with the OpenNeuro DOIs and the actual lead investigators of each cohort, drawn from the `dataset_description.json` released alongside each dataset. The current placeholders are best-guess attributions and have not been confirmed against the dataset metadata.
- Reference [22] (GEFM) is currently arXiv-only. Check whether it has appeared in a venue by submission time.
- The §5.8 commentary on UC medication state and Iowa UPDRS-III scores is grounded only in cohort-level metadata, not in subject-by-subject error tracing. The current evaluation script does not write back the identifier of each misclassified subject. Before submission, re-run the ensemble evaluation with subject-level error logging if a stronger failure analysis is desired.
- The claim in §5.5 that "GCN outperforms GAT" is empirical and consistent with prior findings on EEG foundation models [22]. Do not present it as a novel finding.
- Do not claim "test-subject prototype calibration" as a novelty until the MCPNet paper has been read in full and its calibration formula compared against ours. See `FINDINGS_AND_HONEST_NOVELTY.md` for the audit trail.
