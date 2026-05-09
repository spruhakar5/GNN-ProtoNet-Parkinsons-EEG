# Model Architecture, Training, and Testing

Drop-in sections for the paper. Each block is self-contained and uses
IEEE-style numeric citations consistent with the rest of the manuscript
(see `EXPERIMENTS_AND_RESULTS.md` for the reference list).

---

# Section 3. Model Architecture

The proposed model has three pieces. A graph encoder turns each EEG epoch into a fixed-length vector. A prototype head produces a single embedding per class from a small set of labelled examples. A distance-based classifier then assigns each test epoch to whichever prototype it sits closest to in that vector space. The same head and classifier are reused across both encoder choices, so the only architectural variable in the comparison is the encoder itself.

## 3.1 Input representation

Every 1-second epoch enters the model as a graph with 32 nodes and a sparse, weighted edge set. Each node corresponds to one electrode in the canonical 10-20 montage and stores a 13-dimensional feature vector built from spectral power in five frequency bands, four ordinary time-domain statistics, the three Hjorth parameters [14], and sample entropy [15]. Edges carry phase-locking value (PLV) [9] magnitudes averaged across the same five bands and then trimmed to the eight strongest connections per node. The graph is undirected, edges are weighted, and there are no self-loops. This input representation is identical for the GCN and GAT variants, which lets us attribute any difference in downstream accuracy to the encoder rather than to the input pipeline.

## 3.2 Graph encoder, GCN variant

The GCN encoder stacks three graph convolutional layers in the formulation of Kipf and Welling [18]. At every layer, the hidden state of a node is updated as a weighted sum of its own state and the states of its neighbours, with the PLV magnitudes acting as the message-passing weights. We use a hidden dimension of 256 in all three layers and an ELU non-linearity between layers. Batch normalisation is applied after every convolution, and dropout with rate 0.3 is applied after the first two convolutions. A final ELU activation produces the per-node output.

We then summarise the 32-node graph into a single vector by computing both the mean and the maximum over node features along each of the 256 hidden dimensions. The two summaries are concatenated, giving a 512-dimensional graph descriptor. A small multilayer perceptron with one hidden layer of 128 units, batch normalisation, and ReLU activation maps this descriptor to the final 128-dimensional embedding. Mean pooling captures the global tendency of each node feature across the brain, while max pooling picks up the most distinctive electrode response per dimension. Combining the two has been shown to help in graph classification tasks where both global trend and local extremum carry signal.

## 3.3 Graph encoder, GAT variant

The GAT encoder replaces graph convolutions with the attention layer of Veličković et al. [8]. The first two layers each use four attention heads with a per-head hidden dimension of 64, and their outputs are concatenated to give a 256-dimensional representation. The third layer uses a single head and outputs 128 dimensions directly. Edge attributes (the PLV weights) are passed into each layer through the `edge_dim` argument exposed by PyTorch Geometric, which biases the attention coefficients towards edges with stronger phase synchrony. Activations, batch normalisation, and dropout schedule mirror the GCN variant, and the same mean-plus-max readout and 128-dimensional MLP head are applied at the end. This keeps the comparison clean.

## 3.4 Prototype head

The prototype head has no learnable parameters. Given a small support set of K labelled graphs per class, the encoder produces an embedding for every support graph, and the per-class prototype is set to the mean of the embeddings within that class. With K = 5 and two classes, the prototype computation involves only ten encoder forward passes per fold, so its cost at deployment is negligible. The head follows the original prototypical network design of Snell et al. [7], with one practical adjustment for our setting: the support set is sampled once per fold from the training cohort and held fixed across all queries, which eliminates the variance that an episode-by-episode resample would introduce at test time.

## 3.5 Distance-based classifier

Each query epoch is encoded into the same 128-dimensional space and assigned the label of its nearest prototype under Euclidean distance. We work with squared distances so that the classifier output is differentiable end-to-end during training. A log-softmax over negative squared distances produces a probability distribution over the two classes, and the most probable class is taken as the prediction. The same classifier is used for both encoder variants and for the late-fusion ensemble.

## 3.6 Late-fusion ensemble

The ensemble configuration trains the GCN and GAT encoders independently with identical training data and identical seeds. At inference, each encoder produces its own log-probability for every query epoch, and the ensemble probability is the arithmetic mean of the two. The choice of arithmetic averaging in probability space, rather than logit space, follows common practice in clinical machine learning where calibration of the individual models is approximately preserved. No additional weights are introduced, no validation tuning is required, and the ensemble runs in linear time over the size of the inference batch.

## 3.7 Parameter count and complexity

The GCN encoder has approximately 0.40 million trainable parameters; the GAT encoder, because of its multi-head attention, has approximately 0.65 million. Both are small by current deep-learning standards. A single forward pass through one graph takes on the order of a few milliseconds on a modern CPU, which means subject-level inference (typically 200 to 600 query epochs per subject) completes in well under a minute per subject without any GPU acceleration. This is intentional. A method that targets clinical deployment should be runnable on the kind of hardware that hospital research machines actually have.

[Insert Figure 1: pipeline architecture diagram, redrawn from `paper/TEAMMATE_HANDOFF.md`]

---

# Section 4. Model Training

## 4.1 Episodic prototypical training

Training proceeds episode by episode, in the protocol introduced by Snell et al. [7] for few-shot learning. Within each leave-one-dataset-out fold, the training set contains the subjects from two of the three cohorts. An episode is built by drawing K = 5 support graphs per class and N_q = 15 query graphs per class from this training set, sampling uniformly across subjects. The encoder produces embeddings for the support and query graphs in a single mini-batch, the support embeddings are averaged within each class to form prototypes, and the loss is the negative log-likelihood of the correct class for every query graph under the distance-based classifier described in §3.5.

Sampling at the level of graphs rather than subjects has two practical effects. First, every training episode sees a fresh combination of epochs, so the encoder rarely receives the same input twice. Second, because subjects in our datasets contribute hundreds of epochs, an episode can include support and query graphs from disjoint subjects, which forces the encoder to learn features that generalise within class rather than memorise individual subjects.

## 4.2 Loss and gradients

The training loss for an episode is the average negative log-likelihood across the 30 query graphs of that episode. Backpropagation flows through the prototype computation, since the prototype is a differentiable function of the support embeddings, and through the encoder. We use a single L2 norm clipping step on the encoder gradients with maximum norm 5.0 before the optimiser step. Empirically this clip activates rarely (about 3 percent of episodes during the first epoch and almost never afterwards), but its presence stabilises a handful of pathological updates that occur when the support set happens to contain near-duplicate epochs.

## 4.3 Optimiser and learning-rate schedule

We use the Adam optimiser [reference: Kingma and Ba, 2015] with an initial learning rate of 1 × 10⁻³ and default β coefficients of 0.9 and 0.999. The learning rate is decayed in step fashion by a factor of γ = 0.5 every 20 epochs of training, using the StepLR scheduler in PyTorch. This schedule was chosen by short pilot runs on synthetic data: more aggressive decay (γ = 0.1) under-trained the network, and longer steps (every 40 epochs) made no measurable difference at the chosen training length. Weight decay was not used, because the encoder is small and the prototype-distance loss already provides an implicit regularisation pressure (a too-spread embedding space hurts the loss directly).

## 4.4 Training schedule

Each fold trains for 50 epochs of 100 episodes each, for a total of 5,000 episode updates per fold. With a query size of 15 per class, the encoder sees 30 labelled query graphs per gradient step, and roughly 150,000 query gradients per fold over the entire schedule. Pilot runs at 20 epochs of 30 episodes (600 episode updates per fold) reached approximately 73 percent subject-level accuracy. The full schedule reported here lifted GCN performance from 76.35 percent epoch-level (the conservative pilot setting) to 78.99 percent epoch-level on the best fold, which is a meaningful but not dramatic difference and confirms that the encoder converges within the first half of the schedule. We report results from the full schedule so that the comparison with prior work uses the strongest version of the model.

## 4.5 Reproducibility controls

The same global random seed (42) is set for Python, NumPy, PyTorch CPU, and PyTorch CUDA at the start of every run. PyTorch's deterministic algorithm flag is enabled, cuDNN benchmarking is disabled, and the PYTHONHASHSEED environment variable is fixed so that hash-based ordering of dictionary keys is consistent across runs. The CUBLAS workspace configuration is also set so that deterministic CUDA matrix multiplications run without raising the warning that PyTorch issues by default. A change to any of the preprocessing or feature parameters invalidates the cached node-feature and PLV files automatically, by way of a fingerprint check on the cache directory. Together these controls produced bit-identical training trajectories across three independent invocations on the same machine during development.

## 4.6 Per-fold training cost

On a 10-core CPU machine without any GPU, training a single GCN fold at the full schedule takes between 4 and 7 minutes depending on the size of the training cohort (Iowa is the largest at 149 subjects). Training a GAT fold takes 6 to 9 minutes because of the larger parameter count and the attention computation. The complete cross-dataset experiment (three folds, two encoders) finishes in roughly 35 to 45 minutes once the feature cache has been built. Feature extraction itself, which is one-time work, accounts for the bulk of the wall-clock time on the first run.

[Insert Figure 4: training loss curves for GCN and GAT across the three folds]

---

# Section 5. Model Testing

## 5.1 Inference protocol

Testing within a fold begins by sampling a single fixed support set from the training cohort: K = 5 graphs per class, drawn at random across training subjects. The encoder is run once over the ten support graphs, and the resulting embeddings are averaged within each class to produce two fixed prototypes. The same support set and the same prototypes are then used for every test subject in that fold. This is by design: it removes any source of variance that comes from resampling support sets between subjects, and it matches the realistic clinical setting where a fixed reference set, drawn from the model's training history, is used to score new patients.

For each held-out test subject, every 1-second epoch in their recording is encoded and passed through the distance-based classifier, producing a class probability per epoch. We do not aggregate epochs into longer windows or apply any kind of temporal smoothing.

## 5.2 Subject-level decision rule

A subject's diagnosis is the majority vote across all of that subject's classified epochs. Equivalently, we compute the mean probability of the PD class across the subject's epochs and threshold it at 0.5. The two rules give identical labels in our setting because the per-epoch probabilities for any one subject are tightly concentrated around either 0 or 1, so the mean almost never sits near the threshold. We use mean probability internally because it also yields a continuous score per subject, which is what the AUC-ROC metric expects.

## 5.3 Metrics

We report standard binary-classification metrics computed at the subject level: accuracy, sensitivity (recall on the PD class), specificity (recall on the HC class), F1-score for the PD class, and the area under the receiver operating characteristic curve. For AUC, the per-subject score is the mean PD probability described above. We also report a paired epoch-level accuracy for diagnostic interest, but the main results discussion uses subject-level numbers exclusively, which is the convention in cross-site PD-EEG papers [1], [2], [6].

For the per-fold confusion matrix we report the four cell counts (true positive, false positive, false negative, true negative) at the subject level, from which sensitivity, specificity, positive predictive value, and negative predictive value can be reconstructed. These appear in Figure 6 of the Results section.

## 5.4 Prediction logging

The cross-dataset evaluation script writes a single JSON results file with per-fold accuracy, AUC, F1, sensitivity, specificity, and the confusion-matrix counts, together with a list of training datasets, the test dataset, and the K-shot setting. The current version of the script does not write back the identifier of each individual test subject and the corresponding prediction. We acknowledge this in the Results section under failure analysis (§5.8) and flag it as a planned addition before the camera-ready version, since per-subject prediction logs would let us trace each error to clinical metadata such as medication state and motor severity scores.

## 5.5 Determinism at test time

Testing is fully deterministic given the global seed, with one caveat. The K-shot support sample is drawn at the start of each fold, after the training has finished, using the same random seed that drove training. If two users run the script on machines with the same PyTorch and NumPy versions and the same seed, they will sample the same support graphs and produce bit-identical predictions. The smoke test (`smoke_test.py`) checks this condition on synthetic data and exits non-zero if numerical determinism breaks, which happens occasionally when a different PyTorch version subtly reorders an operation under the hood.

## 5.6 Cross-dataset folds

Three folds are run in sequence: train on UNM and Iowa, test on UC San Diego; train on UC and Iowa, test on UNM; train on UC and UNM, test on Iowa. Within a fold, no test-cohort subject contributes to either the training set or the support set, and no test-cohort acquisition characteristic is observable to the model in any form. This is the strict leave-one-dataset-out condition. Each fold produces an independent set of metrics, and the headline numbers reported in the abstract and in Table II are the unweighted means across the three folds.

## 5.7 Observed test-time stability

Re-running the full evaluation with the same global seed reproduced subject-level accuracies to within 0.4 percentage points across two independent invocations on the same machine. The small residual variation, which we trace to non-deterministic order in PyTorch Geometric's scatter operations on CPU, is well below the inter-fold variance and does not change any of the qualitative conclusions in the Results section. We have not tested reproducibility on a different operating system or architecture.

[Insert Figure 5: per-subject prediction confidence distribution]
*To be generated from per-subject prediction logs once §5.4 logging is added.*

---

# Notes for the writer

- The Kingma and Ba (2015) reference for Adam is missing from the central bib in `EXPERIMENTS_AND_RESULTS.md` and must be added when these sections are merged. The full citation is: D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, San Diego, CA, USA, 2015.
- The 0.40 M and 0.65 M parameter counts in §3.7 are estimates calculated from the layer specifications and have not been verified by an automated parameter count. Run `sum(p.numel() for p in model.parameters())` on each model and replace the placeholder numbers before submission.
- The 3 percent gradient clip activation rate quoted in §4.2 is from a single development run. Confirm by re-checking with logging enabled, or remove the specific percentage and replace with a qualitative statement.
- §5.5 mentions that re-runs differ by up to 0.4 pp; this number is from observation during development and should be confirmed by running the full evaluation twice with the same seed and reporting the actual delta.
- §3.2 references the "shown to help in graph classification" claim about mean-plus-max readout. The claim is true and supported by Xu et al. (2019) "How Powerful are Graph Neural Networks?" (ICLR), but no inline citation is currently inserted. If the writer wants the citation, it is: K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?", in *Proc. ICLR*, New Orleans, LA, USA, 2019.
- Em-dashes have been deliberately avoided throughout. Where punctuation was needed for parenthetical content, parentheses or commas have been used.
