# Video Script — GNN-ProtoNet (Parkinson's Disease EEG)

Target length: ~3 min 30 s. Pacing: ~150 words per minute. Background: GitHub repo at github.com/spruhakar5/GNN-ProtoNet-Parkinsons-EEG.

---

## [0:00 – 0:20] Intro
On-screen: repo root, README visible.

"Hi, I'm Spruha Kar from Delhi Technological University. This is a short walk-through of our work on cross-cohort Parkinson's disease detection from resting-state EEG using graph neural networks. Everything you'll see, the code, the results, and the paper, is in the repository on the screen."

## [0:20 – 0:55] The problem
On-screen: scroll through README "Motivation" section, then briefly show data/raw folder structure.

"Parkinson's disease is usually diagnosed only once visible motor symptoms appear, by which point the underlying neural decline has been going on for years. EEG is appealing as an earlier, quantitative signal, but there is a catch. Models trained at one clinic rarely transfer to another. Different headsets, different demographics, different recording rooms. We tested this directly. Train a graph encoder on two of the three public OpenNeuro Parkinson's EEG cohorts and test on the third, and accuracy collapses toward chance."

## [0:55 – 1:15] Data
On-screen: open data/ folder, point to UC, UNM, Iowa subfolders.

"We used all three publicly available OpenNeuro Parkinson's EEG datasets: UC San Diego, UNM, and Iowa. Together they cover 230 subjects, 141 with Parkinson's and 89 healthy controls. All resting-state, all 64 channels. Pre-processing covers bandpass, notch, and channel harmonisation. That's in src/preprocessing.py."

## [1:15 – 1:50] Method, the graph
On-screen: open src/graph_builder.py and src/features.py.

"Each EEG recording is turned into a graph. The 64 electrodes are nodes. Edges come from the top-k phase-locking values, so the graph captures functional connectivity between regions. Every node carries a feature vector that combines spectral power, time-domain statistics, Hjorth parameters, and sample entropy. The graph construction is in src/graph_builder.py, and the node features are in src/features.py."

## [1:50 – 2:25] Method, the model
On-screen: open src/models/proto_net.py.

"On top of those graphs we train two encoders. One is a graph convolutional network, and the other is a graph attention network. Both are trained episodically under a prototypical-network objective: each training step samples a small support set and a query set, and the encoder learns to place same-class graphs near each other. At inference, we take K labelled epochs from the held-out subject's own recording, and we use them to reposition the class prototypes. That is the calibration step. Finally, the two encoders are combined by late fusion."

## [2:25 – 3:00] Results
On-screen: open results/cross_dataset_full_training.json and ensemble_results.json.

"Evaluation is strict leave-one-dataset-out. Train on two cohorts, test on the third, no subject overlap. With K equals five, the GCN alone reaches 94.47 percent subject-level accuracy and 0.9845 AUC. The GCN plus GAT ensemble reaches 98.22 percent and 0.9961 AUC, getting 226 of 230 subjects right. One healthy control was mis-flagged, three Parkinson's subjects were missed."

## [3:00 – 3:25] Calibration honesty
On-screen: open results/calibration_ablation.json.

"We want to be upfront. Without calibration, both encoders sit close to chance. So the headline number is K-shot supervised personalisation, not zero-shot transfer. K equals one already buys 86.7 percent. By K equals five the curve flattens. Five labelled epochs from a known patient or control is a small ask in clinic."

## [3:25 – 3:40] Efficiency and close
On-screen: scroll back to README, briefly show file list.

"The whole ensemble has 0.28 million parameters, and a single graph runs in 2 milliseconds on CPU. The code, results, and ablations are all in the repository. Thanks for watching."

---

## Delivery notes

- Speak slightly slower on the numbers (94.47, 98.22, 0.9961, 226 of 230).
- The honesty paragraph is the most important moment: pause before "Without calibration, both encoders sit close to chance."
- If you go over 3:30, the trim points are: shorten the Intro, drop the per-fold breakdown, and merge "Method, the graph" with "Method, the model" by skipping the file path mentions.
- Avoid the word "early" if you've already removed it from the title.
- If recording in one take is hard, the seven section headings are natural cut points.
