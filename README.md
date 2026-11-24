🧠 MotorMind — EEG-Based Motor Imagery Classification

A Research-Oriented Study of Left- and Right-Hand Motor Imagery Using Classical Machine Learning and Deep Learning

""Python" (https://img.shields.io/badge/Python-3.10%2B-blue)" (https://www.python.org/)
""PyTorch" (https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)" (https://pytorch.org/)
""MNE" (https://img.shields.io/badge/MNE-EEG%20Processing-purple)" (https://mne.tools/)
""Status" (https://img.shields.io/badge/Status-Ongoing%20Research-orange)" (#10-future-work)

---

📌 1. Project Overview

MotorMind is a research-oriented EEG-based motor imagery classification project that investigates whether electroencephalography (EEG) signals can distinguish between imagined left-hand and right-hand movements.

The project uses the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB) and explores multiple approaches to EEG signal processing and classification, progressing from traditional machine learning methods to deep learning architectures.

The primary objective is to investigate cross-subject motor imagery classification, particularly whether a model trained on multiple participants can generalize to previously unseen participants.

Key Objectives

- EEG dataset inspection, validation and preprocessing.
- Extraction of left-hand and right-hand motor imagery trials.
- Development of conventional 3-channel and 64-channel EEG representations.
- Implementation of classical machine learning baselines.
- Investigation of frequency-specific EEG features.
- Deep learning experimentation using EEGNet Fusion V2.
- Cross-subject evaluation using Leave-One-Subject-Out (LOSO) validation.

«Project Status: Ongoing research prototype. Current experiments focus on offline EEG classification and cross-subject generalization. The system is not yet a validated real-time brain-computer interface.»

---

📂 2. Dataset

The project uses the "PhysioNet EEG Motor Movement/Imagery Dataset" (https://physionet.org/content/eegmmidb/).

The dataset contains EEG recordings from participants performing different motor execution and motor imagery tasks.

For this project, the relevant experimental conditions are:

Event| Description
T0| Resting state
T1| Imagined left-hand movement
T2| Imagined right-hand movement

The classification task focuses on distinguishing T1 from T2.

Dataset Processing Summary

Parameter| Result
Valid EDF recordings| 201
Participants represented| 69
Sampling frequency| 160 Hz
Extracted 3-channel trials| 3,013
Extracted 64-channel trials| 3,013
Classification classes| Left vs Right
Main motor channels| C3, Cz, C4

3-channel representation:

- Left imagery: 1,525 trials
- Right imagery: 1,488 trials

64-channel representation:

- Left imagery: 1,510 trials
- Right imagery: 1,503 trials

One trial was rejected during the 64-channel processing workflow. Nine recordings were identified as corrupt or truncated during dataset processing and auditing.

The 64-channel representation preserves spatial information across the EEG electrode array, allowing experimentation beyond the conventional C3, Cz and C4 configuration.

---

⚙️ 3. EEG Preprocessing Pipeline

A reproducible preprocessing workflow was developed to transform raw EDF recordings into structured EEG trials.

Processing Workflow

Raw EDF Recordings
        |
        v
Dataset Inspection & Validation
        |
        v
Signal Filtering
        |
        v
Motor Imagery Trial Extraction
        |
        v
Channel Selection
   /           \
  v             v
3-Channel     64-Channel
Representation Representation
  \             /
   v           v
Signal Quality Assessment
        |
        v
Dataset Consolidation
        |
        v
Model Training & Evaluation

Processing Stages

1. Raw Data Inspection: Examine EDF files, recording metadata, sampling frequencies, channel information and event annotations.

2. Data Integrity Verification: Identify invalid, incomplete or corrupted recordings and verify trial-level consistency.

3. Signal Filtering: Apply band-pass filtering to isolate frequency components relevant to motor imagery.

4. Trial Extraction: Extract EEG segments corresponding to left-hand and right-hand motor imagery events.

5. Channel Selection: Develop both a conventional three-channel representation and a consolidated 64-channel representation.

6. Signal Quality Assessment: Inspect invalid values, flat signals, recording inconsistencies and trial-level signal characteristics.

7. Dataset Consolidation: Maintain consistent channel ordering and trial-label alignment for subsequent model training.

64-Channel Processing

The consolidated 64-channel representation has the following shape:

(3013, 64, 640)

Where:

- "3013" represents the number of trials.
- "64" represents the EEG channels.
- "640" represents the number of time samples per trial.

The channel ordering was standardized, and the positions of C3, Cz and C4 were verified against the corresponding three-channel representation.

Preprocessing Scripts

- "preprocess_64ch_pipeline.py"
- "preprocessing_pipeline.py"
- "batch_preprocess_pipeline.py"
- "step1_verify_raw_data.py"
- "step2_trial_integrity_audit.py"
- "step3_filter_eeg.py"
- "step4_extract_trials.py"
- "step5_signal_quality.py"

---

🧪 4. Experimental Approaches

The project explores multiple classification approaches to establish baselines and investigate deep learning-based alternatives.

4.1 Common Spatial Patterns + Linear Discriminant Analysis (CSP-LDA)

The initial classification approach combines two conventional machine learning techniques.

Common Spatial Patterns (CSP)

CSP extracts spatial filters that maximize the difference in variance between two EEG classes.

Linear Discriminant Analysis (LDA)

LDA uses the extracted features to classify trials into left-hand or right-hand motor imagery.

This approach provides a conventional machine learning baseline for comparison with subsequent experiments.

Implementation:

- "train_baseline_csp_lda.py"
- "run_csp_lda_random_split.py"

4.2 Filter Bank Common Spatial Patterns (FBCSP)

Frequency-specific experimentation was introduced to investigate the contribution of different EEG frequency bands to motor imagery classification.

The experiments examine:

Frequency Band| Range
Mu| 8–13 Hz
Beta| 13–30 Hz
Combined| 8–30 Hz

FBCSP provides a frequency-oriented classical machine learning comparison against the baseline approach.

Implementation:

- "run_fbcsp_fixed_concatenation.py"
- "run_frequency_band_comparison.py"

4.3 EEGNet Fusion V2

The project subsequently expanded into deep learning experimentation using an EEGNet Fusion V2 architecture.

EEGNet is a compact convolutional neural network architecture designed for EEG-based brain-computer interface applications.

The V2 implementation extends the experimental framework beyond traditional CSP-based feature extraction and classification.

Research objectives:

- Investigate deep learning-based EEG feature learning.
- Explore spatial and temporal information in multichannel EEG recordings.
- Evaluate classification performance across participants.
- Investigate generalization to previously unseen subjects.
- Compare deep learning experiments with classical machine learning baselines.

The implementation includes a dedicated LOSO training and evaluation workflow.

Implementation:

- "run_eegnet_fusion_v2_loso.py"
- "run_eegnet_random_split.py"

«Note: EEGNet Fusion V2 is an ongoing experimental implementation. Its final validated performance and comparison against FBCSP should be reported only after reviewing the corresponding evaluation outputs.»

---

📊 5. Experimental Evaluation

The project uses different evaluation protocols to investigate both within-dataset classification and cross-subject generalization.

5.1 Random Train-Test Evaluation

Random train-test evaluation is used for preliminary model experimentation.

It provides an initial assessment of model learning behaviour and classification performance.

However, because trials from the same participant may occur in different splits, this evaluation does not establish generalization to completely unseen participants.

5.2 Leave-One-Subject-Out (LOSO) Evaluation

LOSO validation is used to investigate cross-subject generalization.

In each iteration:

1. One participant is held out for testing.
2. The remaining participants are used for model training.
3. The trained model is evaluated on the held-out participant.
4. The process is repeated across participants.

This evaluation strategy is particularly relevant to BCI research because EEG signals can vary considerably between individuals.

Evaluation Metrics

The project considers the following metrics:

- Accuracy
- Balanced Accuracy
- ROC-AUC
- Confusion Matrix
- Participant-level classification performance

These metrics help assess both overall classification performance and differences in performance across participants.

---

📈 6. Preliminary Experimental Results

The following results summarize completed experiments. Results from different evaluation protocols should not be treated as directly interchangeable.

6.1 CSP-LDA: Random Split

Metric| Result
Accuracy| 61.19%
Balanced Accuracy| 61.20%
ROC-AUC| 0.6799

6.2 CSP-LDA: LOSO Baseline

Metric| Result
Participants evaluated| 69
Held-out trial predictions| 3,013
Pooled accuracy| 59.61%
Balanced accuracy| 59.61%
ROC-AUC| 0.6424
Majority-class accuracy reference| 50.61%

6.3 EEGNet Fusion: Random Split Experiment

An initial EEGNet Fusion experiment using a random train-validation-test split produced:

Metric| Result
Test accuracy| 53.76%
Best epoch| 11
Training time| Approximately 4,350 seconds

These results represent an initial experiment and should not be interpreted as the final performance of the V2 LOSO implementation.

6.4 FBCSP: LOSO Experiments

Configuration| Accuracy
FBCSP baseline| 59.54%
8–30 Hz reference| 59.61%
Mu band (8–13 Hz)| 59.74%
Beta band (13–30 Hz)| 59.94%

The differences are relatively small. These results should be interpreted cautiously because the frequency-band experiments include an epoch-wise filtering consideration that requires further methodological verification.

Results Interpretation

The experiments demonstrate that motor imagery classification remains challenging when generalizing across different participants.

The classical machine learning baseline provides a useful reference point, while the deep learning experiments establish a foundation for further investigation of learned EEG representations.

The results also highlight the importance of consistent preprocessing, validation protocols and participant-level analysis when evaluating EEG classification systems.

---

🗂️ 7. Repository Structure

The repository contains the following major components:

MotorMind-EEG-BCI/
│
├── docs/
│
├── results/
│
├── download_subset.py
├── batch_preprocess_pipeline.py
├── preprocessing_pipeline.py
├── preprocess_64ch_pipeline.py
│
├── step1_verify_raw_data.py
├── step2_trial_integrity_audit.py
├── step3_filter_eeg.py
├── step4_extract_trials.py
├── step5_signal_quality.py
├── step6_download_dataset.py
│
├── train_baseline_csp_lda.py
├── run_csp_lda_random_split.py
├── run_eegnet_random_split.py
├── run_eegnet_fusion_v2_loso.py
├── run_fbcsp_fixed_concatenation.py
├── run_frequency_band_comparison.py
│
├── verify_consolidated_dataset.py
│
├── requirements-eegnet-loso.txt
├── learning_notes.md
├── model_training_explained.md
└── README.md

Note: This is a logical overview of the major project components. Update it to match the exact files currently present in the repository.

---

🛠️ 8. Technologies and Libraries

Technology| Purpose
Python| Core implementation
MNE-Python| EEG data loading and signal processing
NumPy| Numerical computation
SciPy| Signal processing and filtering
Pandas| Dataset organization and analysis
Scikit-learn| Classical machine learning and evaluation
PyTorch| Deep learning model implementation and training
Matplotlib| Visualization and experimental analysis

---

⚠️ 9. Current Limitations

Several challenges remain in EEG-based motor imagery classification:

1. Inter-subject variability: EEG patterns differ substantially between participants.

2. Signal quality: Recording artifacts, missing channels and incomplete recordings can affect model performance.

3. Generalization: Models that perform well on randomly distributed trials may not generalize equally well to unseen participants.

4. Dataset limitations: The experiments use prerecorded EEG data rather than live EEG acquisition.

5. Deep Learning Performance: The EEGNet Fusion V2 implementation requires further evaluation, tuning and comparison under consistent validation conditions.

6. Experimental Reproducibility: Additional verification of preprocessing, filtering and evaluation procedures is necessary before drawing stronger conclusions.

7. Real-Time Deployment: The current implementation is an offline research prototype and does not yet establish real-time BCI performance.

---

🚀 10. Future Work

The next stages of development include:

- Further evaluation and optimization of EEGNet Fusion V2.
- Consistent comparison of deep learning and classical machine learning approaches.
- Participant-level performance analysis.
- Investigation of frequency-specific EEG representations.
- Improved signal quality assessment and artifact handling.
- Further investigation of cross-subject generalization.
- Reproducibility checks using clean experimental environments.
- Exploration of real-time EEG acquisition and BCI integration as a future direction.

---

📝 11. Conclusion

MotorMind explores EEG-based motor imagery classification through a progressive experimental approach, beginning with classical machine learning methods and extending towards deep learning architectures.

The project establishes a preprocessing and data validation pipeline using the PhysioNet EEG Motor Movement/Imagery Dataset, investigates both three-channel and 64-channel EEG representations, and evaluates classical approaches including CSP-LDA and FBCSP.

The introduction of EEGNet Fusion V2 represents the next stage of experimentation, with particular emphasis on cross-subject generalization using Leave-One-Subject-Out validation.

Rather than focusing exclusively on classification accuracy, the project emphasizes experimental methodology, reproducibility, evaluation across participants and an understanding of the challenges involved in building reliable EEG-based brain-computer interfaces.

MotorMind is an ongoing research and engineering project aimed at developing a deeper understanding of EEG signal processing, motor imagery decoding and machine learning approaches for brain-computer interface applications.

---

📚 12. References

1. "PhysioNet — EEG Motor Movement/Imagery Dataset" (https://physionet.org/content/eegmmidb/)

2. Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. Journal of Neural Engineering. "DOI" (https://doi.org/10.1088/1741-2552/aace8c)

3. "MNE-Python Documentation" (https://mne.tools/)

4. "PyTorch Documentation" (https://pytorch.org/docs/stable/index.html)

---

Repository: "github.com/codeishitech/MotorMind-EEG-BCI" (https://github.com/codeishitech/MotorMind-EEG-BCI)

Project: MotorMind — EEG-Based Motor Imagery Classification
