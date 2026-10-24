# MotorMind - Baseline Report: CSP + LDA with Leave-One-Subject-Out Validation

**Run timestamp:** 2026-10-23 22:29:38  
**Dataset:** `D:\Move-left-or-right\output\epochs\motormind_all_participants_epochs.npz` (SHA-256 `33441d9545a1d21b879954460d75151eec734b0bd86f630d3c866f81095599da`)  
**Validation type:** cross-subject generalization on held-out participants (no per-subject calibration)

*This report was generated automatically from the results of this run.*

## 1. Summary

The dataset contains **3013 trials** from **69 participants** (S001 to S070, 201 participant-run recordings). Class counts: Left_Fist = 1525 (50.61%), Right_Fist = 1488 (49.39%). A majority-class predictor would reach 50.61% accuracy.

- **Pooled accuracy** (3013 held-out predictions): **59.61%**
- **Pooled balanced accuracy:** 59.61%
- **Pooled ROC AUC** (from P(Right_Fist)): 0.6424
- **Pooled macro F1:** 59.60%
- **Mean participant accuracy:** 59.53% (sample SD 11.95%, median 55.56%, range 40.00% to 93.33%, n = 69)
- **Mean participant balanced accuracy:** 59.49% (SD 11.89%, n = 69)

Pooled accuracy weights each participant by their number of trials, whereas the mean participant accuracy weights every participant equally; the two can differ.

## 2. Data and protocol

- Input shape: (3013, 3, 640) (trials, channels, samples)
- Channels: C3, Cz, C4
- Sampling frequency: 160 Hz; epoch duration: 4 s (640 samples)
- Units stored in NPZ: uV
- Preprocessing (as documented by the project, not re-verified here): 8-30 Hz bandpass (not verified by this script)
- Trials per participant: min 30, max 45, mean 43.7

### Validation procedure

Leave-One-Subject-Out cross-validation with 69 folds. In each fold all trials of one participant are held out; CSP and then LDA are fitted only on the remaining participants; the held-out trials are transformed and classified with the fitted models. Each trial receives exactly one held-out prediction. Held-out labels and signals are not used for fitting, normalization, feature selection or hyperparameter tuning. This estimates generalization to **unseen participants**; it does not estimate within-participant (calibrated) performance.

## 3. Model

- Features: `mne.decoding.CSP` with {'n_components': 2, 'reg': None, 'log': True, 'norm_trace': False, 'cov_est': 'concat', 'transform_into': 'average_power'}
- Classifier: `LinearDiscriminantAnalysis` with {'solver': 'svd'}
- No hyperparameters were tuned.

## 4. Pooled held-out results

Confusion matrix (rows = true class, columns = predicted class):

| | Pred Left_Fist (0) | Pred Right_Fist (1) |
| :--- | ---: | ---: |
| **True Left_Fist (0)** | 912 | 613 |
| **True Right_Fist (1)** | 604 | 884 |

| Class | Support | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: | ---: |
| Left_Fist (0) | 1525 | 60.16% | 59.80% | 59.98% |
| Right_Fist (1) | 1488 | 59.05% | 59.41% | 59.23% |

Reference levels: balanced accuracy and ROC AUC of an uninformative binary classifier are 0.5 in expectation. Plain accuracy of an uninformative classifier depends on class balance (majority-class rate here: 50.61%). Macro F1 has no single universal chance level, because it depends on the class balance and on the classifier's prediction rates.

## 5. Per-participant results

| Participant | Trials | L / R | Accuracy | Balanced acc. | Macro F1 | AUC | L->L | L->R | R->L | R->R |
| :---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S001` | 43 | 22 / 21 | 72.09% | 71.75% | 71.33% | 0.797 | 19 | 3 | 9 | 12 |
| `S002` | 45 | 23 / 22 | 60.00% | 59.19% | 53.34% | 0.753 | 22 | 1 | 17 | 5 |
| `S003` | 45 | 23 / 22 | 51.11% | 51.28% | 50.89% | 0.512 | 10 | 13 | 9 | 13 |
| `S004` | 45 | 23 / 22 | 57.78% | 58.70% | 49.74% | 0.848 | 4 | 19 | 0 | 22 |
| `S005` | 45 | 21 / 24 | 57.78% | 55.06% | 47.38% | 0.688 | 3 | 18 | 1 | 23 |
| `S006` | 45 | 24 / 21 | 55.56% | 53.27% | 48.16% | 0.552 | 21 | 3 | 17 | 4 |
| `S007` | 45 | 23 / 22 | 75.56% | 76.09% | 74.29% | 0.968 | 12 | 11 | 0 | 22 |
| `S008` | 45 | 22 / 23 | 57.78% | 57.51% | 57.01% | 0.670 | 10 | 12 | 7 | 16 |
| `S009` | 45 | 24 / 21 | 46.67% | 48.81% | 41.81% | 0.472 | 4 | 20 | 4 | 17 |
| `S010` | 45 | 24 / 21 | 68.89% | 69.64% | 68.75% | 0.817 | 14 | 10 | 4 | 17 |
| `S011` | 45 | 23 / 22 | 51.11% | 52.08% | 40.50% | 0.555 | 2 | 21 | 1 | 21 |
| `S012` | 45 | 21 / 24 | 66.67% | 67.26% | 66.60% | 0.728 | 16 | 5 | 10 | 14 |
| `S013` | 45 | 23 / 22 | 62.22% | 62.15% | 62.15% | 0.727 | 15 | 8 | 9 | 13 |
| `S014` | 45 | 22 / 23 | 51.11% | 50.20% | 40.50% | 0.676 | 2 | 20 | 2 | 21 |
| `S015` | 45 | 23 / 22 | 62.22% | 63.04% | 56.76% | 0.840 | 6 | 17 | 0 | 22 |
| `S016` | 45 | 22 / 23 | 55.56% | 54.94% | 51.51% | 0.528 | 6 | 16 | 4 | 19 |
| `S017` | 45 | 23 / 22 | 51.11% | 50.00% | 33.82% | 0.581 | 23 | 0 | 22 | 0 |
| `S018` | 45 | 22 / 23 | 42.22% | 42.49% | 41.50% | 0.449 | 12 | 10 | 16 | 7 |
| `S019` | 45 | 23 / 22 | 71.11% | 71.64% | 69.61% | 0.777 | 11 | 12 | 1 | 21 |
| `S020` | 30 | 16 / 14 | 53.33% | 53.12% | 53.12% | 0.558 | 9 | 7 | 7 | 7 |
| `S021` | 45 | 24 / 21 | 48.89% | 52.08% | 36.31% | 0.694 | 1 | 23 | 0 | 21 |
| `S022` | 30 | 15 / 15 | 66.67% | 66.67% | 64.11% | 0.671 | 6 | 9 | 1 | 14 |
| `S023` | 30 | 15 / 15 | 53.33% | 53.33% | 40.34% | 0.800 | 1 | 14 | 0 | 15 |
| `S024` | 30 | 14 / 16 | 53.33% | 56.25% | 44.44% | 0.634 | 14 | 0 | 14 | 2 |
| `S026` | 30 | 16 / 14 | 66.67% | 67.86% | 66.06% | 0.638 | 8 | 8 | 2 | 12 |
| `S027` | 30 | 15 / 15 | 46.67% | 46.67% | 45.70% | 0.453 | 5 | 10 | 6 | 9 |
| `S028` | 45 | 22 / 23 | 53.33% | 54.05% | 48.33% | 0.587 | 19 | 3 | 18 | 5 |
| `S029` | 45 | 23 / 22 | 93.33% | 93.38% | 93.33% | 0.984 | 21 | 2 | 1 | 21 |
| `S030` | 45 | 24 / 21 | 53.33% | 50.30% | 38.68% | 0.593 | 23 | 1 | 20 | 1 |
| `S031` | 45 | 22 / 23 | 64.44% | 64.62% | 64.29% | 0.773 | 16 | 6 | 10 | 13 |
| `S032` | 45 | 21 / 24 | 75.56% | 73.81% | 72.94% | 0.879 | 10 | 11 | 0 | 24 |
| `S033` | 45 | 23 / 22 | 68.89% | 69.07% | 68.75% | 0.842 | 14 | 9 | 5 | 17 |
| `S034` | 45 | 21 / 24 | 86.67% | 85.71% | 86.11% | 0.988 | 15 | 6 | 0 | 24 |
| `S035` | 45 | 22 / 23 | 51.11% | 52.17% | 37.50% | 0.632 | 22 | 0 | 22 | 1 |
| `S036` | 45 | 23 / 22 | 40.00% | 40.32% | 38.91% | 0.332 | 6 | 17 | 10 | 12 |
| `S037` | 45 | 23 / 22 | 51.11% | 50.00% | 33.82% | 0.583 | 23 | 0 | 22 | 0 |
| `S038` | 45 | 23 / 22 | 46.67% | 45.75% | 35.10% | 0.413 | 20 | 3 | 21 | 1 |
| `S039` | 45 | 23 / 22 | 48.89% | 49.21% | 47.96% | 0.482 | 8 | 15 | 8 | 14 |
| `S040` | 45 | 21 / 24 | 55.56% | 54.17% | 52.73% | 0.474 | 7 | 14 | 6 | 18 |
| `S041` | 45 | 24 / 21 | 57.78% | 55.36% | 49.74% | 0.613 | 22 | 2 | 17 | 4 |
| `S042` | 45 | 22 / 23 | 86.67% | 86.76% | 86.66% | 0.960 | 20 | 2 | 4 | 19 |
| `S043` | 45 | 23 / 22 | 60.00% | 59.09% | 51.32% | 0.899 | 23 | 0 | 18 | 4 |
| `S044` | 45 | 22 / 23 | 60.00% | 59.58% | 58.33% | 0.792 | 9 | 13 | 5 | 18 |
| `S045` | 45 | 23 / 22 | 53.33% | 52.57% | 46.58% | 0.609 | 20 | 3 | 18 | 4 |
| `S046` | 45 | 23 / 22 | 55.56% | 56.13% | 52.73% | 0.626 | 7 | 16 | 4 | 18 |
| `S047` | 45 | 24 / 21 | 55.56% | 52.98% | 45.91% | 0.615 | 22 | 2 | 18 | 3 |
| `S048` | 45 | 22 / 23 | 75.56% | 75.69% | 75.51% | 0.818 | 18 | 4 | 7 | 16 |
| `S049` | 45 | 23 / 22 | 75.56% | 75.40% | 75.36% | 0.875 | 19 | 4 | 7 | 15 |
| `S050` | 45 | 23 / 22 | 48.89% | 50.00% | 32.84% | 0.601 | 0 | 23 | 0 | 22 |
| `S051` | 45 | 24 / 21 | 57.78% | 56.55% | 55.58% | 0.619 | 18 | 6 | 13 | 8 |
| `S052` | 45 | 24 / 21 | 64.44% | 62.20% | 58.53% | 0.835 | 23 | 1 | 15 | 6 |
| `S053` | 45 | 22 / 23 | 55.56% | 55.93% | 54.45% | 0.557 | 16 | 6 | 14 | 9 |
| `S054` | 45 | 23 / 22 | 55.56% | 56.52% | 45.91% | 0.840 | 3 | 20 | 0 | 22 |
| `S055` | 45 | 23 / 22 | 51.11% | 52.08% | 40.50% | 0.923 | 2 | 21 | 1 | 21 |
| `S056` | 45 | 22 / 23 | 80.00% | 80.14% | 79.96% | 0.866 | 19 | 3 | 6 | 17 |
| `S057` | 45 | 24 / 21 | 57.78% | 58.33% | 57.69% | 0.677 | 12 | 12 | 7 | 14 |
| `S058` | 45 | 24 / 21 | 51.11% | 52.98% | 48.00% | 0.562 | 6 | 18 | 4 | 17 |
| `S059` | 45 | 24 / 21 | 53.33% | 50.00% | 34.78% | 0.365 | 24 | 0 | 21 | 0 |
| `S060` | 45 | 23 / 22 | 86.67% | 86.76% | 86.66% | 0.933 | 19 | 4 | 2 | 20 |
| `S061` | 45 | 22 / 23 | 62.22% | 61.36% | 55.03% | 0.737 | 5 | 17 | 0 | 23 |
| `S062` | 45 | 23 / 22 | 93.33% | 93.48% | 93.32% | 0.990 | 20 | 3 | 0 | 22 |
| `S063` | 45 | 22 / 23 | 53.33% | 54.05% | 48.33% | 0.538 | 19 | 3 | 18 | 5 |
| `S064` | 45 | 23 / 22 | 60.00% | 59.98% | 59.98% | 0.745 | 14 | 9 | 9 | 13 |
| `S065` | 45 | 24 / 21 | 51.11% | 53.87% | 42.97% | 0.704 | 3 | 21 | 1 | 20 |
| `S066` | 45 | 22 / 23 | 44.44% | 44.76% | 43.44% | 0.520 | 13 | 9 | 16 | 7 |
| `S067` | 45 | 22 / 23 | 51.11% | 51.58% | 49.07% | 0.508 | 16 | 6 | 16 | 7 |
| `S068` | 45 | 22 / 23 | 48.89% | 50.00% | 32.84% | 0.585 | 22 | 0 | 23 | 0 |
| `S069` | 45 | 24 / 21 | 53.33% | 52.38% | 51.81% | 0.585 | 16 | 8 | 13 | 8 |
| `S070` | 45 | 22 / 23 | 53.33% | 54.35% | 41.85% | 0.794 | 22 | 0 | 21 | 2 |

(L->R = true Left_Fist predicted as Right_Fist, etc. 'n/a' marks an undefined value.)

Participants with accuracy above / equal to / below 50%: **59 / 0 / 10** of 69. Highest accuracy: `S029` (93.33%); lowest: `S036` (40.00%). Because each participant contributes a limited number of trials, individual accuracies are noisy and a value near 50% is not by itself evidence of non-responsiveness.

Class-specific recall across participants: Left_Fist mean 59.48% (SD 31.48%), Right_Fist mean 59.49% (SD 34.29%).

## 6. Statistical assessment

A one-sided Wilcoxon signed-rank test: per-participant balanced accuracy > 0.5 was performed over n = 64 participants with non-zero difference from 0.5 (W = 1930.00, p = 1.325e-09, alpha = 0.05); the test rejected the null hypothesis that participant balanced accuracy is centred at or below 0.5. Participants are treated as independent samples, but every fold's training set overlaps with the others, so independence is approximate. The test says nothing about individual participants and is not corrected for any other comparison.

No test was run on the pooled trial-level accuracy, because trials from one participant are not independent and a pooled binomial test would overstate certainty.

## 7. Data-quality statement

The script verified array shapes, matching lengths, channel names, sampling frequency, epoch duration, label values and that all EEG values are finite. It did **not** assess artifacts, bad channels, amplitude outliers or inter-participant amplitude differences, and it excluded no trials. The presence of a trial in the NPZ file therefore does not imply that it is artifact-free or otherwise valid.

## 8. Limitations

- Offline analysis of fixed 4 s epochs; not an asynchronous or real-time BCI evaluation.
- Only 3 channels (C3, Cz, C4), so CSP has at most 3 spatial filters to choose from; 2 are used.
- Standard CSP with no regularization or per-participant normalization; amplitude differences between participants may influence the covariance estimates.
- Results measure generalization across participants in this dataset only and may not transfer to other recording setups or populations.
- This is a research baseline, not a clinically validated system.

## 9. Reproducibility

- Python 3.13.0, numpy 2.5.2, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.8.0, mne 1.13.2
- No random components are used in CSP/LDA (svd solver) or in the folds.
- Command: `python train_baseline_csp_lda.py`
