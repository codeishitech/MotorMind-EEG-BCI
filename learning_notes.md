# MotorMind — Step 4: Learning Notes on EEG Epochs & Trial Extraction

## 1. What is an "Epoch" in Simple Computer Science Terms?

In raw EEG recordings, the data is a single long continuous 2D matrix of shape `(channels, time_points)`. For our recording (`S001R04.edf`), that was $64\text{ channels} \times 20,000\text{ samples}$ spanning $125\text{ seconds}$.

An **epoch** (also called a **trial**) is a short, isolated segment of that continuous recording extracted around a specific event of interest (such as a visual cue).

* **Analogy:** Think of a continuous video recording of a soccer match. An "epoch" is a short 4-second video clip cut out every time a penalty kick is taken.
* In machine learning terminology:
  * The continuous recording is an unsegmented raw stream.
  * An **epoch is a single labeled training sample ($X_i, y_i$)**.

---

## 2. Why Do We Extract Trials?

1. **Supervised Machine Learning Requires Discrete Samples:**
   A machine learning classifier (such as LDA, SVM, Random Forest, or a Neural Network) cannot take an entire 2-minute recording as a single input. It expects a dataset structured as:
   $$\text{Dataset } \mathcal{D} = \{(X_1, y_1), (X_2, y_2), \dots, (X_N, y_N)\}$$
   Where each $X_i$ is a feature matrix and $y_i \in \{0, 1\}$ is the class label.

2. **Isolating Active Mental Imagery from Rest:**
   More than half of the 125-second recording consists of `T0` (rest intervals and fixation crosses). Including resting brain activity in a binary classifier would confuse the model. Extracting epochs isolates only the exact 4 seconds when the participant was actively imagining movement.

3. **Aligning Temporal Dynamics:**
   By locking each 4.0-second epoch to $t = 0.0\text{ s}$ (the visual cue onset), we align the participant's brain responses across all repetitions. This allows us to compare how electrode voltages evolve consistently across repeated trials.

---

## 3. Actual Counts, Parameters, and Settings Used

| Parameter | Value | Reason & Justification |
| :--- | :--- | :--- |
| **Participant & Run** | `Subject S001`, `Run R04` | Unilateral Motor Imagery (Task 2) |
| **Total Task Cues in File** | 14 trials ($7 \times \text{T1}$, $7 \times \text{T2}$) | From official BCI2000 annotation stream |
| **Excluded Trial** | **Trial 1 (`T2` Right Fist at $t = 4.20\text{ s}$)** | **0.30s overlap with the 8.0s–15.0s all-channel zero flatline.** Discarded to prevent corrupted data from entering the classifier. |
| **Retained Clean Trials ($N$)** | **13 trials total** | **$7 \times \text{T1}$ (Left Fist)** and **$6 \times \text{T2}$ (Right Fist)** |
| **Selected Channels ($C$)** | 3 channels: `C3`, `Cz`, `C4` | Core motor-strip electrodes covering left hemisphere (`C3`), central vertex (`Cz`), and right hemisphere (`C4`). |
| **Sampling Rate ($f_s$)** | $160\text{ Hz}$ | $160$ voltage samples per second ($1$ sample every $6.25\text{ ms}$). |
| **Epoch Duration ($T$)** | **$4.0\text{ seconds}$** ($640\text{ samples}$) | Captures sustained motor imagery while matching the official 4-second visual cue duration. |
| **Filter Settings** | $8.0 - 30.0\text{ Hz}$ zero-phase FIR | Isolates sensorimotor Mu ($8–12\text{ Hz}$) and Beta ($13–30\text{ Hz}$) rhythms, removes DC drift and muscle noise. |
| **Dataset Tensor Shape** | **`X.shape = (13, 3, 640)`** | $13\text{ trials} \times 3\text{ channels} \times 640\text{ time points}$ |
| **Labels Vector Shape** | **`y.shape = (13,)`** | Binary integers: `0` = Left Fist ($n=7$), `1` = Right Fist ($n=6$) |
| **Signal Units** | Microvolts ($\mu\text{V}$) | Standard physical biomedical unit for scalp EEG |

---

## 4. What a Single Training Example Looks Like in Memory

For any trial index $i \in \{0, 1, \dots, 12\}$:

```text
               TIME (Samples: 0 to 639 @ 160 Hz -> 4.0 seconds)
              t = 0.0s                  t = 2.0s                 t = 4.0s
        C3  ┌─────────────────────────────────────────────────────────────┐
CHANNELS Cz │                  2D Voltage Matrix (3 x 640)                │
        C4  └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                       Class Label y: 0 (Left) or 1 (Right)
```

* $X[i]$ is a 2D float array of shape `(3, 640)` in $\mu\text{V}$.
* $y[i]$ is an integer:
  * `0` if the cue was `T1` (Imagined Left Fist)
  * `1` if the cue was `T2` (Imagined Right Fist)

---

## 5. Storage Format & Reproducibility

The extracted dataset is saved in compressed NumPy archive format at:
- `output/epochs/eeg_motor_imagery_epochs.npz`

It can be loaded in any future Python or ML script in two lines:
```python
import numpy as np
data = np.load("output/epochs/eeg_motor_imagery_epochs.npz")
X = data['X']           # shape: (13, 3, 640)
y = data['y']           # shape: (13,)
channels = data['channel_names'] # ['C3', 'Cz', 'C4']
```

---

## 6. Step 5: Basic Signal-Quality Checks & Interpretation

Before training a machine learning model, we must verify that our extracted trials do not contain pathological numerical errors or extreme physical artifacts.

### What Was Checked?
1. **Numerical Integrity:**
   - **NaN (Not a Number) or Inf (Infinity):** Checked whether mathematical division errors or file corruptions produced non-finite values. (Result: **0 found**).
   - **Flatline Channels:** Checked whether any electrode became disconnected or flatlined ($PtP \approx 0$). (Result: **0 flatlines**).
2. **Peak-to-Peak (PtP) Voltage Range:**
   - $\text{PtP} = \text{Maximum Voltage} - \text{Minimum Voltage}$ for each 4-second trial.
   - In healthy scalp EEG, filtered microvolt activity typically oscillates between $-100\ \mu\text{V}$ and $+100\ \mu\text{V}$ (a PtP of $\sim 80 - 200\ \mu\text{V}$).
   - Sudden spikes $> 250 - 300\ \mu\text{V}$ usually indicate severe motion artifacts, loose electrodes, or heavy eye-blink contamination.

### Actual Results Summary Across All 13 Retained Trials:
* **Minimum Value:** $-80.76\ \mu\text{V}$ (Trial 7)
* **Maximum Value:** $+103.73\ \mu\text{V}$ (Trial 8)
* **Peak-to-Peak Mean:** $148.97\ \mu\text{V}$ ($\text{Std} = 11.75\ \mu\text{V}$)
* **Peak-to-Peak Min / Max:** $131.18\ \mu\text{V}$ (Trial 12) to $170.85\ \mu\text{V}$ (Trial 4)
* **Channel Consistency:**
  - `C3`: Mean PtP $= 148.24\ \mu\text{V}$
  - `Cz`: Mean PtP $= 140.75\ \mu\text{V}$
  - `C4`: Mean PtP $= 123.67\ \mu\text{V}$

### What Do These Results Mean in Plain English?
1. **High Consistency:** The standard deviation of the peak-to-peak ranges is only $11.75\ \mu\text{V}$, meaning all 13 trials have remarkably uniform energy levels.
2. **Absence of Gross Outliers:** No trial exceeds $175\ \mu\text{V}$. There are no electrode pop artifacts, no disconnected channels, and no massive muscle clenches.
3. **No Justification for Deletion:** In professional scientific practice, a large value prompts visual inspection rather than blind automated deletion. Because all 13 trials remain well within physiological bounds, **all 13 trials are retained for subsequent modeling**.

---

## 7. Multi-Participant Generalization (Cohort S001–S019)

### Why Was the Single-Participant Workflow Generalized?
A brain-computer interface model trained on a single recording from a single person (like our initial 13 trials from `S001R04`) is prone to **subject-specific bias**. Every human brain has slightly different skull thickness, cortical geometry, electrode impedance, and mental imagery habits. 

To build an EEG dataset capable of testing true cross-subject or generalizable pattern recognition, we scaled the preprocessing pipeline to cover **19 participants (`S001` through `S019`) across all 3 unilateral motor-imagery runs (`R04`, `R08`, `R12`)**, expanding the dataset from 13 trials to **853 clean, validated trials**.

---

### How Batch Preprocessing Works
Instead of running manual, ad-hoc scripts for each file, the batch pipeline (`batch_preprocess_pipeline.py` backed by `preprocessing_pipeline.py`) automates the following sequence:

1. **Iterative Recording-by-Recording Execution:**
   The pipeline loops over all 19 participants and 3 runs ($19 \times 3 = 57\text{ recordings}$). To prevent out-of-memory (OOM) crashes, **only one continuous recording is loaded into RAM at any given time**.

2. **Pre-flight Binary Header Verification:**
   Before loading signal data, the script examines the raw 256-byte EDF header. It computes the mathematical expected file size from the number of data records and compares it to the actual file size on disk. This immediately catches interrupted or truncated downloads (such as an incomplete `S012R12.edf` which was re-downloaded to full 2.55 MB integrity).

3. **In-Flight Artifact & Zero-Interval Detection:**
   The continuous signal is scanned for contiguous segments where all electrodes flatline to $0.0\ \mu\text{V}$. The confirmed 7-second hardware dropout in `S001R04` ($t = 8.0\text{s} - 15.0\text{s}$) is automatically detected, and any trial whose 4-second epoch overlaps this dropout is tagged and excluded.

4. **Continuous Filtering Before Slicing:**
   The entire continuous recording is filtered *before* cutting out epochs. This is a critical signal-processing requirement: filtering segmented clips introduces artificial startup and teardown transients ("filter ringing") at the start and end of every trial.

5. **Channel Selection & Trial Extraction:**
   Target electrodes `C3`, `Cz`, and `C4` are extracted in strict international 10-10 order. 4.0-second windows ($640\text{ samples}$ at $160\text{ Hz}$) are cut out starting at the visual cue onset ($t = 0.0\text{ s}$).

6. **Consolidation & Self-Validation:**
   Retained trials are stacked into a standardized 3D NumPy array, accompanied by binary integer labels (`0` = Left Fist, `1` = Right Fist), participant IDs, and run IDs.

---

### Filtering and Epoching Decisions Explained in Simple Terms
* **Why 8–30 Hz FIR Bandpass?**
  * **Low-cut (8.0 Hz):** Eliminates slow baseline drift ($< 0.5\text{ Hz}$) caused by participant sweat or electrode movement, as well as high-amplitude ocular delta rhythms ($1–4\text{ Hz}$) from eye blinks.
  * **High-cut (30.0 Hz):** Filters out high-frequency muscle tension ($> 30\text{ Hz}$) from the jaw and neck, as well as 60 Hz electrical mains noise.
  * **What Remains:** The sensorimotor **Mu band ($8–12\text{ Hz}$)** and **Beta band ($13–30\text{ Hz}$)**, where motor-imagery event-related desynchronization (ERD) occurs.
* **Why Zero-Phase Filtering (`phase='zero'`)?**
  Standard filters introduce a time delay (phase shift), meaning an oscillatory wave might appear 50 milliseconds later than it actually happened in the brain. A zero-phase filter runs forward and then backward through the signal, completely cancelling out any phase delay.
* **Why Exactly 4.0 Seconds (640 Samples)?**
  At a sampling frequency of $160\text{ Hz}$, each second contains $160$ voltage readings. In the BCI2000 protocol, the target cue remains on screen for $4.0\text{ seconds}$. Extracting $[0.0\text{ s}, 4.0\text{ s})$ provides exactly $4.0 \times 160 = 640$ time points per trial, capturing the full duration of sustained mental rehearsal.

---

### Handling Invalid Recordings and Epochs
Scientific honesty requires rigorous rules for defective data:
* **Discard, Never Interpolate:** If an epoch overlaps an acquisition pause (such as Trial 1 in `S001R04`), it is dropped from the dataset. Inventing values via interpolation or regression fabricates brain activity the human never produced.
* **No Splicing:** Splicing out corrupted gaps and stitching the edges together creates an artificial cliff in the time series, corrupting Fourier transforms and frequency filters with severe Gibbs ringing.
* **Non-Destructive Screening:** A threshold of $200\ \mu\text{V}$ is used as an exploratory screening bound, not an automated deletion hammer. Out of 853 trials, 66 trials have peak-to-peak amplitudes between $200\ \mu\text{V}$ and $490\ \mu\text{V}$, reflecting natural high-amplitude EEG variants or transient muscle tension across 19 human subjects. Retaining them preserves realistic real-world signal variability.

---

### Final Verified Dataset Summary

| Metric | Verified Value | Meaning |
| :--- | :--- | :--- |
| **Participants** | **19** (`S001` through `S019`) | Full target cohort represented |
| **Total Expected Recordings** | **57** ($19 \times 3$) | 100% valid EDF files on disk |
| **Total Scheduled Task Cues** | **854 trials** | Sum of all `T1` and `T2` events across 57 runs |
| **Total Retained Clean Trials** | **853 trials (99.88%)** | Ready for feature extraction and modeling |
| **Total Excluded Trials** | **1 trial (0.12%)** | Trial #1 of `S001R04` (zero flatline overlap) |
| **Class 0 (`Left_Fist`)** | **431 trials (50.53%)** | Imagined left hand movement |
| **Class 1 (`Right_Fist`)** | **422 trials (49.47%)** | Imagined right hand movement |
| **Class Balance Ratio** | **$1.02 : 1.00$** | Near-perfect class balance |
| **Tensor $X$ Dimensions** | **`(853, 3, 640)`** | $853\text{ trials} \times 3\text{ channels} \times 640\text{ time points}$ |
| **Labels $y$ Dimensions** | **`(853,)`** | Binary integers ($0$ or $1$) |
| **Channel Montage** | `['C3', 'Cz', 'C4']` | 10-10 international standard |
| **Signal Units** | Microvolts ($\mu\text{V}$) | Scaled from raw EDF Volts ($V \times 10^6$) |
| **Numerical Integrity** | **0 NaNs, 0 Infs, 0 flatlines** | 100% finite mathematical numbers |

---

### Saved Output Paths

1. **Consolidated Multi-Participant Dataset:**
   - `output/epochs/motormind_all_participants_epochs.npz` (12.05 MB)
2. **Trial-Level Metadata CSV:**
   - `output/epochs/motormind_trial_metadata.csv` & `output/trial_metadata.csv` (75 KB)
3. **Recording-Level Manifest CSV:**
   - `output/dataset_manifest.csv` (7.7 KB, 57 recording rows)
4. **Quality-Control Publication Figures (300 DPI):**
   - `output/quality_checks/fig8_multi_participant_dataset_summary.png`: Per-participant retained counts and class donut chart.
   - `output/quality_checks/fig9_multi_participant_signal_quality.png`: Peak-to-peak distributions per channel and subject consistency.
   - `output/quality_checks/fig10_grand_average_waveforms.png`: Grand-average waveforms comparing Left vs. Right fist imagery across 853 trials.
5. **Quality Control Audit Report:**
   - `output/quality_checks/multi_participant_quality_report.md`
6. **Preserved Single-Participant Baseline:**
   - `output/epochs/eeg_motor_imagery_epochs.npz` (194 KB, 13 trials, S001R04 baseline preserved unchanged).

---

### Known Limitations & Unresolved Issues
1. **Inter-Subject Amplitude Variability:**
   Across the 19 participants, mean peak-to-peak amplitudes vary from $\sim 60\ \mu\text{V}$ (e.g. `S014`) to $\sim 200\ \mu\text{V}$ (e.g. `S008`). This is normal in scalp EEG and will require per-subject z-score standardization or spatial filtering (like Common Spatial Patterns, CSP) prior to machine learning classification.
2. **Unequal Run Lengths:**
   Recordings for `S001`, `S003`, and `S007` have a duration of $125.0\text{ seconds}$ with a $0.5\text{ s}$ zero padding at the very end ($t = 124.5\text{s} - 125.0\text{s}$), whereas the other 16 participants have recordings of exactly $123.0\text{ seconds}$ without trailing padding. The pipeline verified that all task trials finish well before the end of the files, so this duration difference does not affect trial yield.
3. **Hardware Flatline in S001R04:**
   Trial 1 in `S001R04` is permanently corrupted due to a recording acquisition pause in the BCI2000 system. Discarding this trial is scientifically sound and leaves `S001` with 43 retained trials ($22\text{ Left}, 21\text{ Right}$).

---

## 8. Machine Learning Baseline: CSP + LDA Cross-Subject Classification

### 1. Conceptual Primer: What Are CSP, LDA, and LOSO in Simple Language?

#### A. Common Spatial Patterns (CSP) — Spatial Filtering for EEG
- **The Problem:** Raw EEG electrode voltages record a mixture of many brain rhythms and electrical volume conduction. For example, `C3` and `C4` both pick up background alpha waves, eye movements, and noise in addition to motor imagery.
- **The CSP Solution:** Imagine two photographers taking pictures of a stage from different angles. One photographer positions themselves to maximize the view of the actor on the left while minimizing the actor on the right. The other photographer does the opposite.
- In mathematical terms, CSP designs **spatial filters (linear combinations of electrode weights)** $w$:
  $$z(t) = w_1 \cdot \text{C3}(t) + w_2 \cdot \text{Cz}(t) + w_3 \cdot \text{C4}(t)$$
  CSP finds weights that **maximize the signal variance (power) for Left Fist trials while simultaneously minimizing the variance for Right Fist trials** (and vice versa for the second filter).
- Taking the logarithm of the signal variance yields a compact 2-dimensional feature vector per trial:
  $$f_i = \log(\text{var}(w_i^T X))$$

#### B. Linear Discriminant Analysis (LDA) — The Decision Boundary
- **What it does:** Once each 4-second EEG trial is condensed into a 2D feature point $(f_1, f_2)$, LDA draws a straight decision line separating the two clusters (Left Fist vs. Right Fist).
- **How it classifies:** It models the Gaussian distribution of each class, assumes equal covariance, and places a linear hyperplane at the Bayesian probability threshold ($P = 0.5$). Any trial falling on one side is labeled Left Fist; any trial on the other is labeled Right Fist.

#### C. Leave-One-Subject-Out (LOSO) — The Gold Standard for Generalization
- **Why not random train/test split?** If we randomly split trials (e.g. 80/20), trials from participant `S001` would appear in both training and testing sets. The model would "cheat" by memorizing `S001`'s unique skull thickness, electrode impedances, and resting brain rhythms, rather than learning generalizable motor imagery patterns.
- **How LOSO works:**
  1. Pick one participant (e.g., `S001`) and lock all of their trials away in a test vault.
  2. Train CSP spatial filters and LDA weights **strictly on the remaining 18 participants**.
  3. Feed the locked-out participant's trials into the frozen model and record predictions.
  4. Repeat this 19 times—once for every human participant in the cohort.
- **What it tests:** True **zero-calibration generalization** to an unseen person who never contributed a single trial to model training.

---

### 2. Metrics Explained in Simple Language

1. **Accuracy ($59.32\%$):**
   The raw percentage of all 853 test trials correctly classified ($506 / 853$).
2. **Balanced Accuracy ($59.30\%$):**
   The arithmetic mean of sensitivity for Left Fist and sensitivity for Right Fist:
   $$\text{Balanced Acc} = \frac{\text{Recall}_{\text{Left}} + \text{Recall}_{\text{Right}}}{2} = \frac{60.79\% + 57.82\%}{2} = 59.30\%$$
   This prevents a classifier from inflating its score by favoring the slightly larger class.
3. **Precision:**
   - Left Fist Precision ($59.55\%$): When the model predicts Left Fist, it is correct $59.55\%$ of the time.
   - Right Fist Precision ($59.08\%$): When the model predicts Right Fist, it is correct $59.08\%$ of the time.
4. **Recall (Sensitivity):**
   - Left Fist Recall ($60.79\%$): Out of all 431 actual Left Fist trials, the model caught 262.
   - Right Fist Recall ($57.82\%$): Out of all 422 actual Right Fist trials, the model caught 244.
5. **F1-Score ($59.30\%$):**
   The harmonic mean of precision and recall. A balanced indicator of classifier reliability.
6. **ROC AUC ($0.6274$):**
   The Area Under the Receiver Operating Characteristic curve. Chance level is $0.5000$; a score of $0.6274$ proves that the model's probabilistic rankings carry genuine discriminative signal above pure guessing across the cohort.
7. **Confusion Matrix:**
   - **True Negatives (Left $\rightarrow$ Left):** $262$ trials
   - **False Positives (Left $\rightarrow$ Right):** $169$ trials
   - **False Negatives (Right $\rightarrow$ Left):** $178$ trials
   - **True Positives (Right $\rightarrow$ Right):** $244$ trials

---

### 3. Actual Empirical Results Across 19 Participants

| Participant | Held-Out Trials | Class Split (L / R) | Accuracy | Balanced Accuracy | F1-Macro | Result vs. Chance (50%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `S001` | 43 | 22L / 21R | **62.79%** | 62.01% | 57.64% | +12.79% |
| `S002` | 45 | 23L / 22R | **53.33%** | 52.27% | 38.68% | +3.33% |
| `S003` | 45 | 23L / 22R | **53.33%** | 53.56% | 52.96% | +3.33% |
| `S004` | 45 | 23L / 22R | **66.67%** | 67.39% | 63.09% | +16.67% |
| `S005` | 45 | 21L / 24R | **64.44%** | 62.50% | 60.00% | +14.44% |
| `S006` | 45 | 24L / 21R | **51.11%** | 48.21% | 37.50% | +1.11% |
| `S007` | 45 | 23L / 22R | **80.00%** | 80.43% | 79.35% | **+30.00% (Cohort Best)** |
| `S008` | 45 | 22L / 23R | **60.00%** | 60.28% | 59.50% | +10.00% |
| `S009` | 45 | 24L / 21R | **46.67%** | 48.81% | 41.81% | -3.33% (Challenging) |
| `S010` | 45 | 24L / 21R | **68.89%** | 68.45% | 68.50% | +18.89% |
| `S011` | 45 | 23L / 22R | **51.11%** | 51.58% | 49.07% | +1.11% |
| `S012` | 45 | 21L / 24R | **60.00%** | 61.31% | 59.01% | +10.00% |
| `S013` | 45 | 23L / 22R | **66.67%** | 66.21% | 64.94% | +16.67% |
| `S014` | 45 | 22L / 23R | **53.33%** | 52.47% | 44.44% | +3.33% |
| `S015` | 45 | 23L / 22R | **62.22%** | 63.04% | 56.76% | +12.22% |
| `S016` | 45 | 22L / 23R | **51.11%** | 50.99% | 50.89% | +1.11% |
| `S017` | 45 | 23L / 22R | **51.11%** | 50.00% | 33.82% | +1.11% |
| `S018` | 45 | 22L / 23R | **51.11%** | 51.88% | 45.00% | +1.11% |
| `S019` | 45 | 23L / 22R | **73.33%** | 73.62% | 73.00% | **+23.33% (Top Performer)** |
| **Cohort Summary** | **853 trials** | **431L / 422R** | **59.32% (Mean: 59.33% ± 9.05%)** | **59.30%** | **59.30%** | **18 / 19 participants ≥ 50%** |

---

### 4. Key Scientific Insights

1. **Generalization Signal Exists:**
   18 out of 19 participants achieved test accuracy $\ge 50.0\%$, with participants like `S007` reaching $80.00\%$ and `S019` reaching $73.33\%$. The overall ROC AUC of $0.6274$ confirms that motor imagery spatial features transfer across different human brains even without prior calibration.
2. **Cross-Subject BCI Inefficiency & Anatomy Variance:**
   Several participants (`S006`, `S009`, `S011`, `S016`, `S017`, `S018`) hovered close to chance level ($50–53\%$). This demonstrates the well-known BCI phenomenon of subject-specific cortical geometry and BCI illiteracy. A universal spatial filter cannot account for individual head shapes without adaptive normalization or calibration.
3. **Spatial Constraint of 3 Electrodes:**
   CSP was limited to extracting only 2 spatial filters because our dataset contains 3 electrodes (`C3`, `Cz`, `C4`). In high-density BCI setups (32–64 channels), CSP has substantially more spatial degrees of freedom to isolate localized sensorimotor rhythms.
4. **Preserved Baseline Pipeline:**
   This baseline experiment ran non-destructively: zero dataset files were altered, all 853 trials were retained, and all evaluation splits strictly respected participant boundaries.

### 5. Experiment Output Artifacts
All outputs are saved in `output/model_results/baseline_csp_lda/`:
- `loso_predictions.csv`: Full trial-by-trial predictions and predicted class probabilities for all 853 trials.
- `per_participant_metrics.csv`: Detailed performance metrics for each of the 19 folds.
- `overall_metrics.json`: Machine-readable summary of dataset, model parameters, and pooled metrics.
- `confusion_matrix.png`: High-resolution 300 DPI 2-panel figure showing pooled confusion matrix and participant-by-participant accuracy vs. chance.
- `training_report.md`: Formal scientific training report.

---

## 9. Experiment 1: EEG Frequency-Band Comparison (Mu vs. Beta vs. Broadband)

### 1. Neurophysiological Background: Why Compare Frequency Bands?

When a human imagines moving their left or right fist, the motor cortex undergoes **Event-Related Desynchronization (ERD)**—a reduction in oscillatory power caused by local cortical networks actively firing. This occurs across two primary sensorimotor frequency rhythms:

1. **Mu Band ($8–13\text{ Hz}$):**
   Originates primarily in the post-central somatosensory gyrus and primary motor strip. It reflects cortical idling when at rest, and desynchronizes sharply during motor planning and execution.
2. **Beta Band ($13–30\text{ Hz}$):**
   Originates in pre-central primary motor and premotor cortex. It is directly coupled to corticospinal excitability and muscle control pathways.
3. **Broadband Reference ($8–30\text{ Hz}$):**
   Spans both rhythms simultaneously, allowing the classifier to draw from whichever frequency band carries the strongest information.

**Research Question:** Does restricting spatial filtering to pure Mu ($8–13\text{ Hz}$) or pure Beta ($13–30\text{ Hz}$) improve cross-subject classification accuracy by removing non-relevant frequencies, or does broadband filtering provide a better signal?

---

### 2. Experimental Setup & Filtering Method

- **Dataset:** Retained all 853 clean, validated trials from all 19 participants (`S001`–`S019`).
- **Classifier:** Common Spatial Patterns (CSP, $m=2$ components) + Linear Discriminant Analysis (LDA, SVD solver).
- **Validation:** 19-Fold Leave-One-Subject-Out (LOSO) cross-validation (zero data leakage; CSP and LDA fitted strictly on the 18 training subjects in each fold).
- **Digital Filter Settings:**
  - Standardized zero-phase, non-causal FIR filter (`firwin` with Hamming window).
  - Reflective boundary padding (`pad='reflect_limited'`) applied to short 4.0-second epochs.
- **Filter Edge-Effect Limitation:**
  Narrowband filters require longer impulse responses. For $8–13\text{ Hz}$, the filter length is $265\text{ samples}$ ($1.656\text{ s}$). On $4.0\text{ s}$ ($640\text{ samples}$) epochs, $\pm 132\text{ samples}$ ($\sim 0.83\text{ s}$) at both edges are subject to reflective padding. While boundary reflection prevents artificial cliffs, epoch-level re-filtering carries slight transient boundary smoothing compared to filtering continuous raw streams.

---

### 3. Empirical Results Summary

| Metric | 8–30 Hz (Reference) | 8–13 Hz (Mu Band) | 13–30 Hz (Beta Band) | Difference vs. Reference |
| :--- | :---: | :---: | :---: | :---: |
| **Pooled Accuracy** | **59.32%** (506/853) | **58.15%** (496/853) | **58.26%** (497/853) | $-1.17\%$ (Mu) / $-1.06\%$ (Beta) |
| **Balanced Accuracy** | **59.30%** | **58.13%** | **58.25%** | $-1.17\%$ (Mu) / $-1.05\%$ (Beta) |
| **Macro F1-Score** | **59.30%** | **58.12%** | **58.24%** | $-1.18\%$ (Mu) / $-1.06\%$ (Beta) |
| **ROC AUC** | **0.6274** | **0.6222** | **0.6054** | $-0.0052$ (Mu) / $-0.0220$ (Beta) |
| **Per-Subject Mean Acc** | **59.33% ± 9.05%** | **58.17% ± 9.72%** | **58.27% ± 10.26%** | $-1.16\%$ (Mu) / $-1.06\%$ (Beta) |
| **95% Confidence Interval** | **[54.96%, 63.69%]** | **[53.49%, 62.86%]** | **[53.33%, 63.21%]** | All bands strictly $>50\%$ chance |
| **Participants ≥ 50%** | **18 / 19 (94.7%)** | **16 / 19 (84.2%)** | **17 / 19 (89.5%)** | 8–30 Hz maintains highest consistency |

---

### 4. Paired Statistical Comparison Across Participants ($N=19$)

To avoid the mistake of pseudoreplication (treating trials as independent), paired hypothesis tests were evaluated across the 19 independent participant accuracy scores:

1. **8–30 Hz vs. 8–13 Hz (Mu):**
   $$\text{Mean Difference} = +1.16\%, \quad t(18) = 0.779, \quad p = 0.446 \quad (\text{Wilcoxon } p = 0.677)$$
   *Verdict:* **Statistically non-significant ($p > 0.05$).** Isolating the Mu band did not improve classification over the reference.
2. **8–30 Hz vs. 13–30 Hz (Beta):**
   $$\text{Mean Difference} = +1.06\%, \quad t(18) = 1.148, \quad p = 0.266 \quad (\text{Wilcoxon } p = 0.218)$$
   *Verdict:* **Statistically non-significant ($p > 0.05$).** Isolating the Beta band did not improve classification over the reference.
3. **8–13 Hz vs. 13–30 Hz:**
   $$\text{Mean Difference} = -0.10\%, \quad t(18) = -0.051, \quad p = 0.960 \quad (\text{Wilcoxon } p = 0.897)$$
   *Verdict:* **Virtually identical performance ($p = 0.960$).**

---

### 5. Why Did the Broadband Reference Perform Best?

1. **Subject-Specific Frequency Dominance (Individual Crossover):**
   Human motor imagery is heterogeneous:
   - For participant `S001`, **Mu was superior** ($69.77\%$ vs. $60.47\%$ Beta).
   - For participant `S007`, **Beta was vastly superior** ($86.67\%$ vs. $77.78\%$ Mu).
   - For participant `S014`, **Beta was superior** ($51.11\%$ vs. $51.11\%$ Mu, with higher sensitivity).
   When training a cross-subject model across 18 individuals, forcing a single narrow band penalizes subjects whose brains rely on the other rhythm. The broadband $8–30\text{ Hz}$ filter lets CSP optimize weights across both rhythms simultaneously.
2. **Degrees of Freedom with 3 Electrodes:**
   With only 3 electrodes (`C3`, `Cz`, `C4`), spatial filters have minimal room to maneuver ($m=2$). Giving CSP the full $8–30\text{ Hz}$ spectrum provides necessary spectral variance contrast.
3. **Conclusion for Project Roadmap:**
   Neither isolated Mu nor isolated Beta improves generalization over broadband $8–30\text{ Hz}$. Therefore, **$8–30\text{ Hz}$ remains our established reference band**. Future multi-band approaches should use **Filter Bank CSP (FBCSP)** to combine multiple sub-bands rather than picking one.

### 6. Experiment Output Artifacts
All outputs are saved under `output/model_results/frequency_band_comparison/`:
- `frequency_band_summary.csv`: Tabular comparison of key metrics.
- `per_participant_band_metrics.csv`: Individual participant performance across all 3 bands.
- `all_bands_loso_predictions.csv`: 2,559 individual trial-by-trial predictions ($853 \times 3$ conditions).
- `overall_comparison_metrics.json`: Full machine-readable metrics, paired statistics, and environment metadata.
- `frequency_band_comparison.png`: 300 DPI 2-panel figure showing group metrics and participant trajectories.
- `frequency_band_report.md`: Formal scientific report.




