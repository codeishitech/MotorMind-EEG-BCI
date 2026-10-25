# MotorMind Model-Improvement Research Review

Reviewed 2026-10-25

## Purpose and comparison rules

MotorMind's current task is zero-calibration, cross-participant classification of
imagined left-fist versus right-fist trials. Its current consolidated data contains
3,013 retained trials from 69 participants, using C3/Cz/C4 at 160 Hz, 4-second
epochs, and a documented 8–30 Hz filter. Results from studies with different
participants, channels, epoch windows, labels, or target-subject calibration are
not directly comparable to MotorMind's LOSO scores.

The project source maps PhysioNet T1 to Left_Fist (class 0) and T2 to Right_Fist
(class 1), and selects runs R04, R08, and R12. The official dataset documentation
defines T1/T2 according to task type; the 2024 curation paper's run table identifies
original runs 4, 8, and 12 as unilateral motor imagery (Task 2), confirming that
this project's run selection and label mapping match the intended task.

## Research comparison

| Source | Dataset and task | Channels, epochs, preprocessing | Method and validation | Reported results and relevance |
| --- | --- | --- | --- | --- |
| Roots, Muhammad & Muhammad (2020), [“Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification”](https://doi.org/10.3390/computers9030072) | PhysioNet EEGMMIDB; the abstract reports motor-execution and motor-imagery experiments over 103 subjects. | The accessible abstract does not specify the channel subset, epoch window, or filtering details. | Multi-branch 2D CNN, with branch-specific hyperparameters; reported as cross-subject. The exact split construction and whether it is equivalent to strict participant-held-out LOSO could not be verified from the accessible abstract. The publisher page returned HTTP 403 during this review. | The abstract reports 84.1% for executed and 83.8% for imagined motor actions, with statistically significant improvement over EEGNet, ShallowConvNet, and DeepConvNet; it also reports up to 4× the computational cost of the lowest-cost comparator. This is the most directly relevant neural-network study, but its scores must not be compared directly with MotorMind until the exact task, preprocessing, and split are confirmed from the full article. |
| Chowdhury et al. (2023), [“Enhancing Cross-Subject Motor Imagery Classification in EEG-Based Brain–Computer Interfaces by Using Multi-Branch CNN”](https://doi.org/10.3390/s23187908) | PhysioNet EEGMMIDB; binary left-versus-right executed and imagined movement experiments; 103 subjects after excluding six with annotation issues. | The article describes 64 scalp channels. Its code preprocesses at 160 Hz with a 2–60 Hz bandpass, 60 Hz notch, and artifact removal, then divides each trial into eight 80-sample windows. MotorMind instead uses three channels, 8–30 Hz, and the full 4-second trial. | Five-branch EEGNet Fusion V2. The article states a random 70/10/20 sample split across all subjects, not subject-held-out LOSO. The released code also differs from the article: it performs ordered (`shuffle=False`) splitting and uses Adam learning rate 0.001, while the article text states 0.00001. | Reports 89.6% for executed and 87.8% for imagined activity with the selected five-branch model. Since the stated split does not hold out participants, these figures do not estimate strict cross-participant generalization and are not a direct target for MotorMind. The architecture is nevertheless a candidate to adapt and evaluate under MotorMind's strict LOSO protocol. |
| Shuqfa, Lakas & Belkacem (2024), [“Increasing accessibility to a large brain-computer interface dataset: Curation of physionet EEG motor movement/imagery dataset for decoding and classification”](https://doi.org/10.1016/j.dib.2024.110181) | Curation of EEGMMIDB; 103 subjects retained after excluding six subjects with inconsistent recordings. It covers four motor-imagery and four motor-execution tasks. The paper's task table identifies original runs 4, 8, and 12 as unilateral left-versus-right-fist imagery. | Original recordings: 64 EEG channels, 160 Hz. Task runs contain repeated relaxation and task trials; the paper describes roughly four-second trial intervals. A figure illustrates 8–30 Hz filtering, but this is a data-curation paper rather than a model benchmark. | No classifier validation protocol or classification score is reported as an experiment in this paper. | Useful for verifying the dataset/task/run interpretation and documenting EEGMMIDB's incomplete/inconsistent recordings. It does not establish an expected accuracy for MotorMind. The curated IDs are renumbered after exclusions, so they should not be assumed to match original PhysioNet participant IDs without consulting the paper's mapping. |
| Xu, Huang & Lan (2021), [“Selective Cross-Subject Transfer Learning Based on Riemannian Tangent Space for Motor Imagery Brain-Computer Interface”](https://doi.org/10.3389/fnins.2021.779231) | Two MI datasets, neither EEGMMIDB: MI1 has 52 healthy participants performing left- versus right-hand imagery; MI2 is BCI Competition IV Dataset 1, with seven participants performing left-hand versus foot imagery. | MI1: 64 channels, 512 Hz; common-average reference, third-order Butterworth 8–30 Hz, epochs 0.5–2.5 s after cue; 200 or 240 trials per person. MI2: 59 channels, 100 Hz; 8–30 Hz, 50th-order FIR, epochs 0.5–3.5 s after cue, 100 trials per class. The authors select 27 and 29 sensorimotor channels for MI1 and MI2. | Covariance matrices are aligned using a Riemannian reference, mapped to tangent-space vectors, and source subjects are selected with sequential forward floating search. An sLDA classifier is used. This is calibrated transfer learning: target-subject trials are randomly split into labeled training and unlabeled testing sets (10–40% labeled), repeated ten times; source labels are also assumed available. | On MI1's seven preselected “good” target subjects, at 40% labeled target data, mean accuracy was 87.60% (CSP-LDA), 96.75% (sSCSTL), and 98.51% (semi-supervised ssSCSTL). The seven were selected based on CSP performance, so these figures are not representative of all 52 participants. They are not zero-calibration LOSO results and cannot be used as a direct MotorMind target. |

## What the evidence means for MotorMind

1. **Keep the evaluation target explicit.** MotorMind's primary target should remain
   zero-calibration LOSO: every trial for a participant is held out together. Any
   method that uses target-participant signals or labels for alignment, tuning, or
   training must be reported separately as a calibrated/adaptive experiment.
2. **Repair the comparison protocol before interpreting frequency results.** The
   existing frequency-band report is an older 19-participant/853-trial experiment.
   It reports 59.32% for its 8–30 Hz reference, 58.15% for 8–13 Hz, and 58.26% for
   13–30 Hz, with no statistically significant paired differences. These results
   are not comparable to the 69-participant baseline. In addition, its sub-band
   filters are applied to already broadband-filtered, short epochs, creating edge
   transients; a future comparison should use identical folds and apply all
   conditions consistently, preferably on continuous recordings before epoching.
3. **FBCSP is a reasonable next classical experiment.** A small, predeclared bank of
   sub-bands could let CSP extract complementary spatial features. Any feature
   selection or weighting must be fitted only within grouped inner validation on
   the training participants of each outer LOSO fold.
4. **Riemannian covariance features are promising but should be adapted carefully.**
   The tangent-space representation turns covariance matrices into vectors that a
   conventional classifier can use. For a strict zero-calibration test, all
   reference/alignment parameters must be learned from outer-training participants
   only. Alignment that uses the held-out participant's signals is a separate
   transductive/adaptation condition and must be labeled accordingly.
5. **Compare classical classifiers on the same features and folds.** LDA, SVM, and
   logistic regression can be compared with a limited, documented, participant-
   grouped inner search. Do not select the winner using outer-test results and then
   report the same outer score as an unbiased final estimate.
6. **Treat EEGNet Fusion V2 as an architecture candidate, not an accuracy promise.**
   Its published EEGMMIDB score uses a random sample split and differs from
   MotorMind's channels and preprocessing. Any reproduction for this project should
   retain strict participant-held-out LOSO, use validation participants drawn only
   from each outer training cohort, and report the paper's split and learning-rate
   discrepancies. The model's greater capacity also makes overfitting and training
   cost important to monitor.

## Project-specific caveats

- Current saved results are 59.61% pooled accuracy for the full-cohort baseline
  and 59.81% for the CSP-3/LDA-shrinkage candidate. The difference is exploratory,
  small, and not statistically persuasive; the candidate was selected after
  comparing configurations on the same cohort.
- The consolidated dataset represents 69 of 70 configured participants. Nine of
  the 210 requested EDF recordings failed validation, including all three runs
  for S025; the remaining six failed runs belong to S020, S022–S024, S026, and
  S027. These exclusions should be reported in future experiments.
- The literature scores above have different cohorts, class definitions, channel
  montages, epoch windows, and/or calibration. They are context, not a leaderboard
  against MotorMind.

## Sources

- PhysioNet, [EEG Motor Movement/Imagery Dataset, version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/).
- Roots, K., Muhammad, Y., & Muhammad, N. (2020). [Fusion Convolutional Neural
  Network for Cross-Subject EEG Motor Imagery Classification](https://doi.org/10.3390/computers9030072).
  *Computers, 9*(3), 72. The publisher page was inaccessible to this review; its
  abstract and bibliographic metadata were checked via Crossref and Semantic Scholar.
- Chowdhury, R. R., et al. (2023). [Enhancing Cross-Subject Motor Imagery
  Classification in EEG-Based Brain–Computer Interfaces by Using Multi-Branch
  CNN](https://doi.org/10.3390/s23187908). *Sensors, 23*(18), 7908. Full text:
  [PMC10536894](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536894/); implementation:
  [EEGNet-Fusion-V2](https://github.com/radia-rayan-chowdhury/EEGNet-Fusion-V2).
- Shuqfa, Z., Lakas, A., & Belkacem, A. N. (2024). [Increasing accessibility to a
  large brain-computer interface dataset: Curation of physionet EEG motor
  movement/imagery dataset for decoding and classification](https://doi.org/10.1016/j.dib.2024.110181).
  *Data in Brief, 54*, 110181. Full text: [PMC10998040](https://pmc.ncbi.nlm.nih.gov/articles/PMC10998040/).
- Xu, Y., Huang, X., & Lan, Q. (2021). [Selective Cross-Subject Transfer Learning
  Based on Riemannian Tangent Space for Motor Imagery Brain-Computer
  Interface](https://doi.org/10.3389/fnins.2021.779231).
  *Frontiers in Neuroscience, 15*, 779231. Full text:
  [PMC8595943](https://pmc.ncbi.nlm.nih.gov/articles/PMC8595943/).
