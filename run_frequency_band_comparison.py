"""
MotorMind — Experiment 1: EEG Frequency-Band Comparison (Mu vs. Beta vs. Broadband)
===================================================================================
Investigates whether isolating specific sensorimotor oscillatory rhythms improves
cross-subject classification of imagined Left Fist vs. Right Fist compared to the
8–30 Hz baseline.

Conditions Evaluated:
  1. 8–30 Hz (Broadband Reference): Sensorimotor Mu + Beta rhythms (existing reference).
  2. 8–13 Hz (Mu / Sensorimotor Alpha Band): Rolandic mu rhythm desynchronization.
  3. 13–30 Hz (Beta Band): Motor cortical beta rhythm desynchronization.

Methodology:
  - Standardized CSP (n_components=2, log=True) + LDA (solver='svd') across all conditions.
  - Strict participant-held-out Leave-One-Subject-Out (LOSO) cross-validation.
  - Paired parametric (paired t-test) and non-parametric (Wilcoxon signed-rank) testing.
  - Documented edge-effect analysis of epoch-level digital filtering.

Outputs (saved under output/model_results/frequency_band_comparison/):
  - frequency_band_summary.csv
  - overall_comparison_metrics.json
  - per_participant_band_metrics.csv
  - all_bands_loso_predictions.csv
  - frequency_band_comparison.png
  - frequency_band_report.md

Command-line options:
  --dataset  Select the input NPZ.
  --output   Select a separate output directory; non-empty directories are protected.
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

import mne
from mne.filter import filter_data
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Suppress verbose MNE logging
mne.set_log_level("WARNING")

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "output", "epochs", "motormind_all_participants_epochs.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "model_results", "frequency_band_comparison")


def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset at {path}")
    with np.load(path, allow_pickle=False) as data:
        return {
            "X": data["X"],
            "y": data["y"],
            "participant_ids": data["participant_ids"],
            "run_ids": data["run_ids"],
            "channel_names": list(data["channel_names"]),
            "sfreq": float(data["sfreq"]),
            "epoch_duration_sec": float(data["epoch_duration_sec"]),
            "units": str(data["units"])
        }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: str) -> str:
    absolute_path = os.path.abspath(path)
    try:
        return os.path.relpath(absolute_path, BASE_DIR)
    except ValueError:
        return absolute_path


def filter_epochs(X: np.ndarray, sfreq: float, l_freq: float, h_freq: float, band_name: str):
    """
    Applies a zero-phase windowed sinc FIR bandpass filter across all epochs.
    """
    print(f"Applying zero-phase FIR bandpass [{l_freq:.1f} – {h_freq:.1f} Hz] for condition '{band_name}'...")
    X_filtered = filter_data(
        data=X.copy(),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        pad="reflect_limited",
        verbose=False
    )
    return X_filtered


def run_loso_for_band(X_band: np.ndarray, y: np.ndarray, participant_ids: np.ndarray, run_ids: np.ndarray, band_key: str):
    """
    Executes one held-out fold per participant for a frequency-band representation.
    """
    unique_subjs = sorted(list(np.unique(participant_ids)))
    n_subjs = len(unique_subjs)

    trial_records = []
    participant_records = []

    for fold_idx, test_subj in enumerate(unique_subjs, start=1):
        test_mask = (participant_ids == test_subj)
        train_mask = ~test_mask

        X_train, y_train = X_band[train_mask], y[train_mask]
        X_test, y_test = X_band[test_mask], y[test_mask]

        test_indices = np.where(test_mask)[0]
        test_run_ids = run_ids[test_mask]

        # 1. Fit CSP strictly on training data
        csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train, y_train)

        # 2. Fit LDA strictly on training CSP features
        lda = LinearDiscriminantAnalysis(solver="svd")
        lda.fit(X_train_csp, y_train)

        # 3. Transform test trials and predict
        X_test_csp = csp.transform(X_test)
        y_pred = lda.predict(X_test_csp)
        y_prob = lda.predict_proba(X_test_csp)

        # Record trial predictions
        for i, global_idx in enumerate(test_indices):
            trial_records.append({
                "band_condition": band_key,
                "trial_index": int(global_idx),
                "participant_id": str(test_subj),
                "run_id": str(test_run_ids[i]),
                "true_label": int(y_test[i]),
                "predicted_label": int(y_pred[i]),
                "prob_left": float(y_prob[i, 0]),
                "prob_right": float(y_prob[i, 1]),
                "is_correct": bool(y_pred[i] == y_test[i])
            })

        # Calculate fold metrics
        fold_acc = float(accuracy_score(y_test, y_pred))
        fold_bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        fold_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        fold_auc = float(roc_auc_score(y_test, y_prob[:, 1])) if len(np.unique(y_test)) == 2 else 0.5
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        participant_records.append({
            "band_condition": band_key,
            "participant_id": str(test_subj),
            "n_trials": len(y_test),
            "accuracy": fold_acc,
            "balanced_accuracy": fold_bal_acc,
            "f1_macro": fold_f1,
            "roc_auc": fold_auc,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        })

    df_trials = pd.DataFrame(trial_records)
    df_part = pd.DataFrame(participant_records)

    # Sort trials by global index
    df_trials.sort_values(by="trial_index", inplace=True)
    df_trials.reset_index(drop=True, inplace=True)

    # Pooled metrics
    y_true_all = df_trials["true_label"].values
    y_pred_all = df_trials["predicted_label"].values
    prob_r_all = df_trials["prob_right"].values

    pooled_acc = float(accuracy_score(y_true_all, y_pred_all))
    pooled_bal_acc = float(balanced_accuracy_score(y_true_all, y_pred_all))
    pooled_f1 = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    pooled_prec = float(precision_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    pooled_rec = float(recall_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    pooled_auc = float(roc_auc_score(y_true_all, prob_r_all))

    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    tn_a, fp_a, fn_a, tp_a = cm_all.ravel()

    # Participant statistics
    accs = df_part["accuracy"].values
    bal_accs = df_part["balanced_accuracy"].values
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs, ddof=1))
    sem_acc = float(std_acc / np.sqrt(len(accs)))
    ci95_acc = stats.t.interval(0.95, df=len(accs)-1, loc=mean_acc, scale=sem_acc)

    t_stat_05, p_val_05 = stats.ttest_1samp(accs, 0.50, alternative="greater")
    w_stat_05, p_val_w05 = stats.wilcoxon(accs - 0.50, alternative="greater")

    metrics_summary = {
        "band_key": band_key,
        "pooled_accuracy": pooled_acc,
        "pooled_balanced_accuracy": pooled_bal_acc,
        "pooled_macro_f1": pooled_f1,
        "pooled_macro_precision": pooled_prec,
        "pooled_macro_recall": pooled_rec,
        "pooled_roc_auc": pooled_auc,
        "confusion_matrix": {
            "tn": int(tn_a),
            "fp": int(fp_a),
            "fn": int(fn_a),
            "tp": int(tp_a),
            "matrix": [[int(tn_a), int(fp_a)], [int(fn_a), int(tp_a)]]
        },
        "participant_distribution": {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "sem_accuracy": sem_acc,
            "median_accuracy": float(np.median(accs)),
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
            "ci95_accuracy": [float(ci95_acc[0]), float(ci95_acc[1])],
            "mean_balanced_accuracy": float(np.mean(bal_accs)),
            "std_balanced_accuracy": float(np.std(bal_accs, ddof=1)),
            "above_chance_count": int(np.sum(accs >= 0.50))
        },
        "significance_vs_chance": {
            "t_statistic": float(t_stat_05),
            "t_p_value_one_sided": float(p_val_05),
            "wilcoxon_stat": float(w_stat_05),
            "wilcoxon_p_value_one_sided": float(p_val_w05)
        }
    }

    return df_trials, df_part, metrics_summary


def compute_comparisons(band_results: dict):
    """
    Computes paired statistical comparisons across held-out participants.
    """
    ref_accs = band_results["8-30_Hz_ref"]["df_part"]["accuracy"].values
    mu_accs = band_results["8-13_Hz_mu"]["df_part"]["accuracy"].values
    beta_accs = band_results["13-30_Hz_beta"]["df_part"]["accuracy"].values

    comparisons = {}

    # Ref vs Mu
    diff_mu = ref_accs - mu_accs
    t_mu, p_mu = stats.ttest_rel(ref_accs, mu_accs)
    w_mu, pw_mu = stats.wilcoxon(diff_mu)
    comparisons["ref_vs_mu"] = {
        "mean_diff_accuracy": float(np.mean(diff_mu)),
        "t_stat": float(t_mu),
        "p_value_two_sided": float(p_mu),
        "wilcoxon_stat": float(w_mu),
        "wilcoxon_p_value": float(pw_mu),
        "significant_p05": bool(p_mu < 0.05)
    }

    # Ref vs Beta
    diff_beta = ref_accs - beta_accs
    t_b, p_b = stats.ttest_rel(ref_accs, beta_accs)
    w_b, pw_b = stats.wilcoxon(diff_beta)
    comparisons["ref_vs_beta"] = {
        "mean_diff_accuracy": float(np.mean(diff_beta)),
        "t_stat": float(t_b),
        "p_value_two_sided": float(p_b),
        "wilcoxon_stat": float(w_b),
        "wilcoxon_p_value": float(pw_b),
        "significant_p05": bool(p_b < 0.05)
    }

    # Mu vs Beta
    diff_mb = mu_accs - beta_accs
    t_mb, p_mb = stats.ttest_rel(mu_accs, beta_accs)
    w_mb, pw_mb = stats.wilcoxon(diff_mb)
    comparisons["mu_vs_beta"] = {
        "mean_diff_accuracy": float(np.mean(diff_mb)),
        "t_stat": float(t_mb),
        "p_value_two_sided": float(p_mb),
        "wilcoxon_stat": float(w_mb),
        "wilcoxon_p_value": float(pw_mb),
        "significant_p05": bool(p_mb < 0.05)
    }

    return comparisons


def validate_loso_predictions(
    band_results: dict,
    participant_ids: np.ndarray,
    n_trials: int,
) -> None:
    expected_indices = set(range(n_trials))
    expected_participants = set(map(str, np.unique(participant_ids)))
    for band_key, result in band_results.items():
        trial_predictions = result["df_trials"]
        participant_metrics = result["df_part"]
        actual_indices = set(trial_predictions["trial_index"].astype(int))
        actual_participants = set(participant_metrics["participant_id"].astype(str))
        if len(trial_predictions) != n_trials or actual_indices != expected_indices:
            raise RuntimeError(
                f"{band_key} must contain exactly one held-out prediction for each "
                f"of {n_trials} input trials."
            )
        if actual_participants != expected_participants:
            raise RuntimeError(
                f"{band_key} participant folds do not match the input cohort."
            )
        if participant_metrics["participant_id"].astype(str).duplicated().any():
            raise RuntimeError(f"{band_key} contains duplicate participant folds.")
        expected_trial_participants = np.asarray(participant_ids).astype(str)[
            trial_predictions["trial_index"].to_numpy(dtype=int)
        ]
        actual_trial_participants = trial_predictions[
            "participant_id"
        ].astype(str).to_numpy()
        if not np.array_equal(actual_trial_participants, expected_trial_participants):
            raise RuntimeError(
                f"{band_key} contains trial predictions assigned to the wrong "
                "held-out participant."
            )


def generate_visualizations(band_results: dict, output_path: str):
    """
    Generates a publication-quality 2-panel figure:
      Panel A: Group-level metrics comparison (Accuracy, Balanced Acc, ROC AUC)
      Panel B: Per-participant accuracy across the three bands
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Panel A: Bar Chart of Key Metrics
    ax0 = axes[0]
    conditions = ["8–30 Hz\n(Reference)", "8–13 Hz\n(Mu Band)", "13–30 Hz\n(Beta Band)"]
    keys = ["8-30_Hz_ref", "8-13_Hz_mu", "13-30_Hz_beta"]
    n_trials = len(band_results["8-30_Hz_ref"]["df_trials"])
    n_subjects = len(band_results["8-30_Hz_ref"]["df_part"])

    accs = [band_results[k]["summary"]["pooled_accuracy"] * 100 for k in keys]
    bal_accs = [band_results[k]["summary"]["pooled_balanced_accuracy"] * 100 for k in keys]
    aucs = [band_results[k]["summary"]["pooled_roc_auc"] * 100 for k in keys]

    x = np.arange(len(conditions))
    width = 0.25

    rects1 = ax0.bar(x - width, accs, width, label="Accuracy", color="#2b5c8f", edgecolor="black")
    rects2 = ax0.bar(x, bal_accs, width, label="Balanced Acc", color="#3498db", edgecolor="black")
    rects3 = ax0.bar(x + width, aucs, width, label="ROC AUC (×100)", color="#e67e22", edgecolor="black")

    ax0.axhline(50.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Chance Level (50.0%)")
    ax0.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax0.set_title(
        f"A. Pooled Generalization Across Frequency Bands\n"
        f"(N = {n_trials:,} trials, {n_subjects} participants)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels(conditions, fontsize=11, fontweight="bold")
    ax0.set_ylim(0, 85)
    ax0.legend(loc="upper right", frameon=True, fontsize=10)
    ax0.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotations
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            h = bar.get_height()
            ax0.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel B: Subject Accuracy Trajectories across conditions
    ax1 = axes[1]
    ref_subjs = band_results["8-30_Hz_ref"]["df_part"]["accuracy"].values * 100
    mu_subjs = band_results["8-13_Hz_mu"]["df_part"]["accuracy"].values * 100
    beta_subjs = band_results["13-30_Hz_beta"]["df_part"]["accuracy"].values * 100

    subj_ids = band_results["8-30_Hz_ref"]["df_part"]["participant_id"].values

    for i in range(len(subj_ids)):
        ax1.plot([0, 1, 2], [ref_subjs[i], mu_subjs[i], beta_subjs[i]], marker="o", color="#7f8c8d", alpha=0.4, linewidth=1.2)

    # Mean trajectory in bold
    mean_traj = [np.mean(ref_subjs), np.mean(mu_subjs), np.mean(beta_subjs)]
    ax1.plot([0, 1, 2], mean_traj, marker="s", markersize=8, color="#c0392b", linewidth=3.0, label="Cohort Mean")

    ax1.axhline(50.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Chance Level (50.0%)")
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(conditions, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Participant Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("B. Individual Participant Accuracy Trajectories\n(Showing Inter-Subject Variability)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visualization saved to: {output_path}")


def generate_report(
    band_results: dict,
    comparisons: dict,
    output_path: str,
    dataset_path: str,
    dataset_hash: str,
    dataset_metadata: dict,
):
    """
    Generates a formal scientific markdown report detailing Experiment 1.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    c_mu = comparisons["ref_vs_mu"]
    c_b = comparisons["ref_vs_beta"]
    c_mb = comparisons["mu_vs_beta"]

    ref_trials = band_results["8-30_Hz_ref"]["df_trials"]
    participant_ids = sorted(
        band_results["8-30_Hz_ref"]["df_part"]["participant_id"].astype(str).unique()
    )
    n_trials = len(ref_trials)
    n_subjects = len(participant_ids)
    class_counts = ref_trials["true_label"].value_counts().to_dict()
    left_count = int(class_counts.get(0, 0))
    right_count = int(class_counts.get(1, 0))
    participant_list = ", ".join(participant_ids)
    dataset_display = display_path(dataset_path)
    output_display = display_path(output_path)

    metric_rows = [
        ("Pooled Accuracy", "pooled_accuracy", "fraction"),
        ("Balanced Accuracy", "pooled_balanced_accuracy", "fraction"),
        ("Macro Precision", "pooled_macro_precision", "fraction"),
        ("Macro Recall", "pooled_macro_recall", "fraction"),
        ("Macro F1-Score", "pooled_macro_f1", "fraction"),
        ("ROC AUC", "pooled_roc_auc", "auc"),
        ("Per-Participant Mean Acc", "participant_mean", "distribution"),
        ("Per-Participant 95% CI", "participant_ci", "ci"),
        ("Above Chance (>= 50%)", "above_chance", "count"),
    ]
    condition_keys = ["8-30_Hz_ref", "8-13_Hz_mu", "13-30_Hz_beta"]
    condition_labels = [
        "8–30 Hz Reference",
        "8–13 Hz Mu",
        "13–30 Hz Beta",
    ]

    def metric_value(condition_key, metric_key, display_type):
        summary = band_results[condition_key]["summary"]
        distribution = summary["participant_distribution"]
        if metric_key == "pooled_accuracy":
            cm = summary["confusion_matrix"]
            correct = cm["tn"] + cm["tp"]
            return f"{summary[metric_key] * 100:.2f}% ({correct}/{n_trials})"
        if display_type == "fraction":
            return f"{summary[metric_key] * 100:.2f}%"
        if display_type == "auc":
            return f"{summary[metric_key]:.4f}"
        if display_type == "distribution":
            return (
                f"{distribution['mean_accuracy'] * 100:.2f}% ± "
                f"{distribution['std_accuracy'] * 100:.2f}%"
            )
        if display_type == "ci":
            low, high = distribution["ci95_accuracy"]
            return f"[{low * 100:.2f}%, {high * 100:.2f}%]"
        if display_type == "count":
            count = distribution["above_chance_count"]
            return f"{count} / {n_subjects} ({count / n_subjects * 100:.1f}%)"
        raise ValueError(f"Unsupported report metric format: {display_type}")

    metric_table = (
        "| Metric | 8–30 Hz (Reference) | 8–13 Hz (Mu) | "
        "13–30 Hz (Beta) | Best Condition |\n"
        "| :--- | :---: | :---: | :---: | :---: |\n"
    )
    for label, metric_key, display_type in metric_rows:
        if metric_key in ("participant_mean", "participant_ci", "above_chance"):
            values = [
                metric_value(key, metric_key, display_type)
                for key in condition_keys
            ]
            if metric_key == "participant_ci":
                raw_values = [
                    -(
                        band_results[key]["summary"]["participant_distribution"]
                        ["ci95_accuracy"][1]
                        - band_results[key]["summary"]["participant_distribution"]
                        ["ci95_accuracy"][0]
                    )
                    for key in condition_keys
                ]
            else:
                distribution_key = (
                    "mean_accuracy"
                    if metric_key == "participant_mean"
                    else "above_chance_count"
                )
                raw_values = [
                    band_results[key]["summary"]["participant_distribution"][
                        distribution_key
                    ]
                    for key in condition_keys
                ]
        else:
            values = [
                metric_value(key, metric_key, display_type)
                for key in condition_keys
            ]
            raw_values = [
                band_results[key]["summary"][metric_key]
                for key in condition_keys
            ]
        best_index = int(np.argmax(raw_values))
        best_label = condition_labels[best_index]
        formatted_values = [
            f"**{value}**" if index == best_index else value
            for index, value in enumerate(values)
        ]
        metric_table += (
            f"| **{label}** | {formatted_values[0]} | {formatted_values[1]} | "
            f"{formatted_values[2]} | **{best_label}** |\n"
        )

    comparison_specs = [
        ("8–30 Hz vs. 8–13 Hz (Mu)", c_mu),
        ("8–30 Hz vs. 13–30 Hz (Beta)", c_b),
        ("8–13 Hz vs. 13–30 Hz", c_mb),
    ]
    comparison_table = (
        "| Comparison | Mean Accuracy Difference | Paired t-test ($t, p$) | "
        "Wilcoxon Signed-Rank ($W, p$) | Significant by t-test ($\\alpha=0.05$)? |\n"
        "| :--- | :---: | :---: | :---: | :---: |\n"
    )
    for label, result in comparison_specs:
        significance = "Yes" if result["significant_p05"] else "No"
        comparison_table += (
            f"| **{label}** | {result['mean_diff_accuracy'] * 100:+.2f}% | "
            f"$t={result['t_stat']:.4f}, p={result['p_value_two_sided']:.4g}$ | "
            f"$W={result['wilcoxon_stat']:.4g}, "
            f"p={result['wilcoxon_p_value']:.4g}$ | **{significance}** |\n"
        )

    best_pooled_index = int(np.argmax([
        band_results[key]["summary"]["pooled_accuracy"]
        for key in condition_keys
    ]))
    best_pooled_label = condition_labels[best_pooled_index]

    report_content = f"""# MotorMind — Experiment 1: EEG Frequency-Band Comparison Report

**Date of Execution:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Project Workspace:** `D:\\Move-left-or-right`  
**Evaluation Scheme:** {n_subjects}-Fold Leave-One-Subject-Out (LOSO) Cross-Validation<br>
**Feature Extractor:** Common Spatial Patterns (CSP, 2 components, log-variance power)  
**Classifier:** Linear Discriminant Analysis (LDA, solver='svd')  
**Input Shape:** {n_trials:,} epochs × {len(dataset_metadata["channel_names"])} channels × {dataset_metadata["X"].shape[-1]} samples<br>
**Sampling Rate / Epoch Duration / Units:** {dataset_metadata["sfreq"]:g} Hz / {dataset_metadata["epoch_duration_sec"]:g} s / {dataset_metadata["units"]}<br>
**Cohort Scope:** {n_subjects} retained participants ({participant_list}); {n_trials:,} trials<br>
**Class Counts:** Left fist: {left_count:,}; right fist: {right_count:,}<br>
**Dataset:** `{dataset_display}`<br>
**Dataset SHA-256:** `{dataset_hash}`<br>
**Output Location:** `{output_display}`

---

## 1. Objective & Scientific Hypothesis

The objective of Experiment 1 is to evaluate whether isolating specific sensorimotor oscillatory rhythms improves cross-subject generalization for binary motor-imagery classification (Left Fist vs. Right Fist).

### Neurophysiological Rationale:
In cognitive neuroscience, voluntary imagined limb movement induces **Event-Related Desynchronization (ERD)**—a power decrease in sensorimotor rhythms over the contralateral motor cortex:
1. **Mu Band ($8–13\\text{{ Hz}}$):** Sensorimotor rhythm associated with cortical activation.
2. **Beta Band ($13–30\\text{{ Hz}}$):** Motor cortical beta rhythm, closely tied to pyramidal tract corticospinal excitability and movement preparation.
3. **Broadband Sensorimotor Band ($8–30\\text{{ Hz}}$):** The baseline reference spanning both Mu and Beta rhythms.

**Hypothesis:** Does restricting the spatial filter to either pure Mu ($8–13\\text{{ Hz}}$) or pure Beta ($13–30\\text{{ Hz}}$) increase signal-to-noise ratio and cross-subject generalization compared to the broadband $8–30\\text{{ Hz}}$ baseline?

---

## 2. Experimental Methodology & Filter Settings

### Filter Implementation & Settings:
Filtering uses MNE's zero-phase FIR implementation (`firwin`, Hamming window,
`reflect_limited` padding) at {dataset_metadata["sfreq"]:g} Hz. The 8–30 Hz reference
is the existing broadband-filtered input. The Mu and Beta conditions apply
additional 8–13 Hz and 13–30 Hz filters to the already-epoched input, separately
for each {dataset_metadata["epoch_duration_sec"]:g}-second epoch.

### Methodological Limitation: Filter Edge Effects on Epochs
Filtering short 4-second epochs introduces edge transients. Reflective padding
reduces boundary discontinuities but does not make epoch-wise filtering equivalent
to filtering the original continuous recordings. In particular, narrow-band Mu
filtering can be sensitive to edge effects; interpret the band comparison with
this limitation in mind.

---

## 3. Empirical Results & Performance Comparison

### Group-Level Performance Summary ({n_trials:,} Trials, {n_subjects} LOSO Folds):

{metric_table}

---

## 4. Paired Inferential Statistical Comparison

Paired parametric and non-parametric tests compare held-out participant accuracy
scores ($N = {n_subjects}$). These tests are exploratory; the three pairwise
comparisons are not adjusted for multiple testing.

### Statistical Hypotheses:
$H_0$: There is no difference in participant accuracy between conditions ($d = 0$).  
$H_1$: A significant difference exists between conditions ($d \\neq 0$).

{comparison_table}

### Scientific Interpretation of Tests:
1. The highest pooled-accuracy condition was **{best_pooled_label}**.
2. The table reports paired t-test and Wilcoxon p-values; a non-significant result
   is not evidence that the conditions are equivalent.
3. Pooled scores and participant-averaged scores answer different questions and
   are both reported to make cohort-size variation visible.

---

## 5. In-Depth Discussion & Neurophysiological Interpretation

### 1. Inter-Participant Variability
- **Inter-Subject Rhythm Dominance:** Human motor imagery is heterogeneous. Some individuals display robust Mu ERD with weak Beta modulation; others exhibit pronounced Beta ERD with minimal Mu rhythm change. A broadband filter ($8–30\\text{{ Hz}}$) allows CSP spatial filters to simultaneously draw variance contrast from whichever rhythm is most prominent in a given participant.
- **Spectral Degrees of Freedom:** The montage contains {len(dataset_metadata["channel_names"])} channels ({", ".join(dataset_metadata["channel_names"])}). Narrow-band filtering removes spectral content that may be useful for some participants.

### 2. Subject-Level Crossover Effects
Participant-level accuracies and all held-out predictions are in the accompanying
CSV files. Per-participant differences should be interpreted alongside the
unequal number of trials contributed by each participant.

### 3. Filter Edge Considerations
The narrow-band filters can have longer effective impulse responses than the
broadband input filter. Exact edge behavior depends on the filter design and epoch
length; this experiment does not estimate performance on continuous-stream-filtered
data.

---

## 6. Documented Limitations & Project Boundaries

1. **No Data Leakage:** The fixed filtering transforms each epoch independently; CSP and LDA are fitted strictly on training participants within each fold.
2. **No Hyperparameter Tuning on Test Folds:** Filter cutoffs were chosen *a priori* based on standard neuroscience definitions ($8–13\\text{{ Hz}}$ and $13–30\\text{{ Hz}}$). No tuning was conducted to optimize test performance.
3. **Preserved Baseline & Consolidated Data:** The original `motormind_all_participants_epochs.npz` and baseline directory `baseline_csp_lda/` were left completely intact.
4. **Honest Significance Reporting:** Statistical results are reported as computed; no improvement is claimed solely from the highest point estimate.

---

## 7. Conclusions & Next Experimental Recommendations

1. **Conclusion:** {best_pooled_label} had the highest pooled accuracy in this run. Whether the conditions differ reliably should be judged from the paired tests above, not point estimates alone.
2. **Next Step — Filter Bank CSP (FBCSP):** Rather than choosing *either* Mu *or* Beta, a multi-band Filter Bank CSP approach can extract spatial filters across multiple sub-bands simultaneously and use feature selection to dynamically pick the best features per subject.
3. **Next Step — Per-Subject Standardization:** Inter-subject amplitude discrepancies remain a primary bottleneck across all frequency bands.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare three EEG frequency-band conditions with participant-held-out LOSO."
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_PATH,
        help="Input consolidated epochs NPZ (default: the project's consolidated dataset).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help="Output directory. Non-empty directories are never overwritten.",
    )
    args = parser.parse_args()
    dataset_path = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output)

    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")
        if os.listdir(output_dir):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose a new --output path to preserve existing results."
            )

    print("=" * 80)
    print("MotorMind — Experiment 1: EEG Frequency-Band Comparison")
    print("=" * 80)

    # 1. Load dataset
    ds = load_dataset(dataset_path)
    X_raw = ds["X"]
    y = ds["y"]
    p_ids = ds["participant_ids"]
    r_ids = ds["run_ids"]
    sfreq = ds["sfreq"]
    dataset_hash = sha256_file(dataset_path)
    n_subjects = len(np.unique(p_ids))
    print(
        f"Dataset: {dataset_path}\n"
        f"SHA-256: {dataset_hash}\n"
        f"Loaded {len(y):,} trials from {n_subjects} participants."
    )

    # 2. Filter data for the three conditions
    # Condition 1: 8-30 Hz (Reference broadband - already in X)
    X_ref = X_raw.copy()

    # Condition 2: 8-13 Hz (Mu band)
    X_mu = filter_epochs(X_raw, sfreq, 8.0, 13.0, "8–13 Hz (Mu)")

    # Condition 3: 13-30 Hz (Beta band)
    X_beta = filter_epochs(X_raw, sfreq, 13.0, 30.0, "13–30 Hz (Beta)")

    bands_data = {
        "8-30_Hz_ref": {"name": "8–30 Hz (Reference)", "X": X_ref},
        "8-13_Hz_mu": {"name": "8–13 Hz (Mu Band)", "X": X_mu},
        "13-30_Hz_beta": {"name": "13–30 Hz (Beta Band)", "X": X_beta}
    }

    band_results = {}
    all_trials_records = []
    all_part_records = []
    summary_rows = []

    print("\nExecuting LOSO evaluations across all conditions...")
    print("-" * 80)

    for b_key, b_info in bands_data.items():
        print(f"\n>>> Running {n_subjects}-Fold LOSO for: {b_info['name']}")
        df_tr, df_p, summ = run_loso_for_band(b_info["X"], y, p_ids, r_ids, b_key)

        band_results[b_key] = {
            "name": b_info["name"],
            "df_trials": df_tr,
            "df_part": df_p,
            "summary": summ
        }

        all_trials_records.append(df_tr)
        all_part_records.append(df_p)

        dist = summ["participant_distribution"]
        summary_rows.append({
            "band_condition": b_key,
            "band_name": b_info["name"],
            "pooled_accuracy": summ["pooled_accuracy"],
            "pooled_balanced_accuracy": summ["pooled_balanced_accuracy"],
            "pooled_macro_precision": summ["pooled_macro_precision"],
            "pooled_macro_recall": summ["pooled_macro_recall"],
            "pooled_macro_f1": summ["pooled_macro_f1"],
            "pooled_roc_auc": summ["pooled_roc_auc"],
            "mean_subject_accuracy": dist["mean_accuracy"],
            "std_subject_accuracy": dist["std_accuracy"],
            "sem_subject_accuracy": dist["sem_accuracy"],
            "ci95_low": dist["ci95_accuracy"][0],
            "ci95_high": dist["ci95_accuracy"][1],
            "above_chance_subjects": dist["above_chance_count"]
        })

        print(f"Done. Pooled Acc: {summ['pooled_accuracy']*100:.2f}%, Bal Acc: {summ['pooled_balanced_accuracy']*100:.2f}%, Mean Subj Acc: {dist['mean_accuracy']*100:.2f}% ± {dist['std_accuracy']*100:.2f}%")

    validate_loso_predictions(band_results, p_ids, len(y))
    print(
        f"\nValidated exactly {len(y):,} unique held-out predictions per condition "
        f"across {n_subjects} participants."
    )

    # 3. Compute paired inferential statistics
    comparisons = compute_comparisons(band_results)

    # 4. Save results to output directory
    os.makedirs(output_dir, exist_ok=True)

    # Summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "frequency_band_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary CSV to: {summary_csv_path}")

    # Participant metrics CSV
    df_all_parts = pd.concat(all_part_records, ignore_index=True)
    part_csv_path = os.path.join(output_dir, "per_participant_band_metrics.csv")
    df_all_parts.to_csv(part_csv_path, index=False)
    print(f"Saved participant metrics to: {part_csv_path}")

    # Trial predictions CSV
    df_all_trials = pd.concat(all_trials_records, ignore_index=True)
    trials_csv_path = os.path.join(output_dir, "all_bands_loso_predictions.csv")
    df_all_trials.to_csv(trials_csv_path, index=False)
    print(f"Saved trial predictions to: {trials_csv_path}")

    # Overall JSON
    overall_json_data = {
        "experiment_name": "frequency_band_comparison",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "path": dataset_path,
            "sha256": dataset_hash,
            "n_trials": int(len(y)),
            "n_participants": int(n_subjects),
            "class_counts": {
                "left_fist": int(np.sum(y == 0)),
                "right_fist": int(np.sum(y == 1)),
            },
            "channel_names": ds["channel_names"],
            "sampling_rate_hz": ds["sfreq"],
            "epoch_duration_sec": ds["epoch_duration_sec"],
            "units": ds["units"],
        },
        "evaluation": {
            "scheme": "Leave-One-Subject-Out",
            "folds": int(n_subjects),
            "csp_components": 2,
            "csp_log": True,
            "lda_solver": "svd",
            "frequency_conditions_hz": {
                "8-30_Hz_ref": "existing broadband-filtered input",
                "8-13_Hz_mu": [8.0, 13.0],
                "13-30_Hz_beta": [13.0, 30.0],
            },
            "epochwise_filtering_caveat": (
                "Mu and Beta are filtered separately on already-epoched data; "
                "this is not equivalent to filtering continuous recordings."
            ),
        },
        "conditions_evaluated": {k: b["summary"] for k, b in band_results.items()},
        "paired_comparisons": comparisons,
        "environment": {
            "mne": mne.__version__,
            "python": sys.version.split()[0]
        }
    }
    json_path = os.path.join(output_dir, "overall_comparison_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_json_data, f, indent=2)
    print(f"Saved comparison JSON to: {json_path}")

    # Plot
    plot_path = os.path.join(output_dir, "frequency_band_comparison.png")
    generate_visualizations(band_results, plot_path)

    # Report
    report_path = os.path.join(output_dir, "frequency_band_report.md")
    generate_report(
        band_results,
        comparisons,
        report_path,
        dataset_path,
        dataset_hash,
        ds,
    )

    print("\n" + "=" * 80)
    print("Experiment 1 Complete! All artifacts generated successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
