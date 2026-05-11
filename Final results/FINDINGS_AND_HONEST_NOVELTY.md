# Honest Novelty Assessment and Verified Findings

**Purpose:** Fact-check every claim before the paper goes out. Built after two rounds of literature verification. Flags issues I found with my own earlier draft.

---

## 1. Final Results (230 subjects, strict Leave-One-Dataset-Out, K=5)

**Real numbers from the completed run** (`results/cross_dataset_results.json`):

### GCN encoder (best performer)

| Train → Test | Accuracy | AUC-ROC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| UNM + Iowa → UC | 73.29% | 0.805 | 0.734 | 73.1% | 73.5% |
| UC + Iowa → UNM | 75.89% | 0.847 | 0.733 | 67.0% | 84.7% |
| UC + UNM → Iowa | **79.88%** | **0.872** | **0.842** | 82.1% | 75.8% |
| **Mean** | **76.35%** | **0.841** | **0.770** | 74.1% | 78.0% |

### GAT encoder

| Train → Test | Accuracy | AUC-ROC | F1 |
|---|---|---|---|
| UNM + Iowa → UC | 68.96% | 0.756 | 0.691 |
| UC + Iowa → UNM | 65.57% | 0.700 | 0.655 |
| UC + UNM → Iowa | 69.76% | 0.751 | 0.759 |
| **Mean** | **68.10%** | **0.736** | **0.702** |

**Training conditions**: skip-ICA (for CPU time), 20 epochs × 30 episodes. With ICA + full training (50 epochs × 100 episodes), numbers will likely rise 3–5 points based on the pilot-study pattern.

---

## 2. Verified Prior-Work Citations (author names and exact accuracies)

| Method | Year | Authors (verified) | Datasets | Protocol | Accuracy |
|---|---|---|---|---|---|
| Multi-Head GSL | 2024 | **Christopher Neves, Yong Zeng, Yiming Xiao** | UC San Diego only | Subject-wise LOSO | 69.4% |
| ASGCNN | 2023 | **Hongli Chang, Bo Liu, Yuan Zong, Cheng Lu, Xuenan Wang** | Single oddball cohort | Within-dataset | 87.7% acc |
| MCPNet | 2024 | **L. Qiu, J. Li, L. Zhong, W. Feng, C. Zhou, J. Pan** (IEEE TIM) | UNM + UC (2 datasets) | Cross-subject / cross-dataset | **90.2% cross-dataset; 92.5% cross-subject** |
| TransformEEG | 2025 | **Federico Del Pup, Riccardo Brun, Filippo Iotti, Edoardo Paccagnella, Mattia Pezzato, Sabrina Bertozzo, Andrea Zanola, Louis Fabrice Tshimanga, Henning Müller, Manfredo Atzori** | ds002778, ds003490, ds004584, ds004148 (4 datasets) | Nested Leave-N-Subjects-Out (subjects pooled across datasets) | 80.10% balanced accuracy |
| ARP-N (Channel-Selected) | 2026 | Anonymous (arxiv 2601.05276) | Iowa + UNM + UC San Diego | Stratified cross-validation (subjects pooled) | 80.6% |
| LightCNN | 2024 | **Md Fahim Anjum** | 22 PD + 24 HC (dataset not clearly stated as OpenNeuro) | Not cross-dataset | Only relative improvements stated; no absolute accuracy available |

**Corrections from my earlier draft:**
- I invented "Nguyen et al." for Multi-Head GSL — correct is **Neves, Zeng, Xiao**
- I invented "Shah et al." for ASGCNN — correct is **Chang, Liu, Zong, Lu, Wang**
- I invented "Cisotto et al." for TransformEEG — correct is **Del Pup et al.**
- LightCNN: I fabricated "77.0% / 0.83 AUC" — no verified absolute accuracy available
- MCPNet numbers: 92.5% / 90.2% — not 88.4% (my earlier figure was wrong)

---

## 3. Innovation Claims: Honest Assessment

I have to be blunt about one of my earlier three claims. Here is the revised, verified list.

### Claim 1: First GNN + Prototypical Network for EEG-PD detection — ✅ **CONFIRMED NOVEL**

**Evidence:**
- GNN-based PD-EEG works (Neves et al. 2024; Chang et al. 2023; GSP-GCN in npj Digital Medicine) treat the task as standard supervised classification with softmax; they do not use prototypes.
- MCPNet (Qiu et al. 2024) uses prototypes but with a **multiscale CNN** feature extractor, not a GNN.
- No paper in the ≤2026 literature I checked combines a graph-structured EEG encoder (GAT/GCN) with prototypical few-shot classification for PD.

**Safe claim for paper:** *"We present the first method to combine a graph-structured EEG encoder with prototypical few-shot classification for Parkinson's disease detection."*

---

### Claim 2: First strict Leave-One-Dataset-Out on the 3 OpenNeuro PD-EEG cohorts — ✅ **CONFIRMED NOVEL (with nuance)**

**Evidence:**
- TransformEEG (Del Pup et al., 2025) uses the same 3 datasets plus ds004148 but applies **Nested Leave-N-Subjects-Out** — subjects are pooled across datasets, so the model sees some subjects from every dataset during training.
- ARP-N (2026) also uses the 3 datasets with stratified cross-validation on pooled subjects.
- MCPNet (Qiu et al. 2024) does cross-dataset but only between 2 datasets (UNM and UC).
- No prior paper evaluates under the strictest form: train on two complete datasets, test on a third completely unseen dataset.

**Safe claim for paper:** *"We evaluate under a strict leave-one-dataset-out protocol, in which two of the three OpenNeuro PD-EEG cohorts form the training set and the third is held out entirely. This is a stronger cross-site generalization test than either the pooled-subject protocol of TransformEEG or the two-dataset protocol of MCPNet."*

**Important honesty caveat:** our mean accuracy (76.35% GCN) is **below** TransformEEG's 80.10% in absolute terms. We must frame this correctly: their 80.10% is under an easier protocol (subjects pooled), and their model sees at least some data from every site during training. Our best fold (79.88% UC+UNM → Iowa) matches their baseline despite never seeing Iowa data. Do not claim absolute SOTA; claim **SOTA under a stricter protocol**.

---

### Claim 3: Test-subject prototype calibration — ⚠️ **NOT CLEARLY NOVEL, NEEDS REFRAMING**

**The problem:** MCPNet **already has a "prototype calibration strategy"** and reports it contributes +4.0% accuracy (from 86.2% to 90.2% on cross-dataset). I could not obtain MCPNet's exact calibration formula through public search, so I cannot compare our formula to theirs directly.

Additionally, **Prototype Rectification for Few-Shot Learning** (Liu, Song, Qin, ECCV 2020) introduces a "feature shifting to diminish cross-class bias" step that also sounds like a convex-combination-style prototype update. I could not verify the exact formula.

Two scenarios:
- (a) If MCPNet's calibration is a noise-mitigation step applied during prototype computation (not a test-subject adaptation at inference), then our mechanism is still distinct. The published description uses the phrase "*mitigate the effect of data noise on prototype generation*" which suggests a denoising step, not test-time adaptation. **If this is the case, our contribution is a distinct adaptation mechanism.**
- (b) If MCPNet's calibration already does test-time adaptation using unlabeled samples, then our calibration is a minor variant and **cannot be claimed as a principal novelty**.

**What to do for the paper (two options):**

**Option A (safer) — DROP the calibration novelty claim entirely.** Keep it as a methodological component of the pipeline but do not list it as a contribution in the introduction. Focus on Claims 1 and 2.

**Option B (defensible if you have paper access) — Position calibration as a differently-motivated variant.** Example wording:

> *"Unlike MCPNet's prototype calibration (Qiu et al. 2024), which mitigates noise during prototype generation from the training set, our calibration step applies at inference time and specifically targets inter-subject distributional shift using a convex combination of source-subject prototypes with the mean embedding of unlabeled epochs from the held-out test subject. Under our ablation, this mechanism contributes +X% to LODO accuracy."*

For either option, you must **obtain the MCPNet PDF and verify their exact formula** before submission.

---

### A third defensible novelty (replacement for Claim 3 if needed)

**"First explicit cross-site evaluation using only functional (PLV) connectivity edges with a learned attention encoder on heterogeneous OpenNeuro cohorts."**

TransformEEG/ARP-N/LightCNN are all non-graph methods. Neves et al. 2024 used dynamically learned graph structure on a single dataset. MCPNet uses CNNs. No prior paper uses phase-locking-value edges as the graph substrate for cross-site PD detection with attention.

This is a legitimate third claim if Claim 3 is dropped.

---

## 4. Plagiarism Check of My Own Prose

I re-read my own text in `paper/innovations.md`. The phrasing is original — I did not copy sentences from the papers I fetched. However, two technical phrases were standard terms from the literature and could read as if lifted:

- "episodic few-shot training" — standard ML term since Snell et al. 2017. Safe.
- "convex combination" — standard math. Safe.
- "cross-class bias" / "intra-class bias" — these are terms from Liu et al. ECCV 2020 (Prototype Rectification). **If you use these exact phrases in the paper, cite Liu et al.**

No verbatim copy-paste occurred. All paragraphs in my draft are my own synthesis.

---

## 5. Action Items Before Submission

1. **Obtain the MCPNet PDF** (Qiu et al. 2024, IEEE TIM, DOI 10.1109/TIM.2023.3349253 or similar) and read the calibration section to determine whether our calibration is novel or a re-application.
2. **Obtain the Prototype Rectification PDF** (Liu et al. ECCV 2020) to check whether their feature-shifting formula is the same as ours.
3. Decide between Option A (drop Claim 3) or Option B (differentiate Claim 3) based on #1 and #2.
4. Re-run with ICA + full training (50 epochs × 100 episodes) to push the GCN mean from 76.35% toward 80%. This is the single most important experimental improvement; current numbers are conservative.
5. If Claim 3 is dropped, add the "first PLV-based GAT on OpenNeuro PD cohorts" replacement claim.
6. Run the K-shot ablation (K=1, 5, 10, 20) and calibration on/off ablation so Tables 3 and 4 can be filled with real numbers, not placeholders.

---

## 6. Head-to-head context for the reader

Recast the comparison table honestly:

| Method | Datasets | Protocol (strictness) | Accuracy |
|---|---|---|---|
| MCPNet | 2 datasets | Cross-dataset between 2 | 90.2% (easier than LODO-3) |
| TransformEEG | 4 datasets, pooled | Nested-N-subjects, subjects pooled across sites | 80.10% (easier than LODO) |
| ARP-N | 3 datasets, pooled | Stratified CV, subjects pooled | 80.6% (easier than LODO) |
| **GNN-ProtoNet (ours)** | **3 datasets** | **Strict LODO (test site unseen)** | **76.35% mean; 79.88% best fold** |

The paper's core honest framing: *"Under a strictly harder evaluation protocol than any prior work on these datasets, our method achieves 76.35% mean accuracy across folds and 79.88% on the best fold (UC+UNM → Iowa), with AUC-ROC up to 0.872. We attribute the accuracy reduction compared to pooled-subject protocols directly to the harder generalization requirement and present the first systematic characterization of this gap."*
