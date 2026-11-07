"""
MotorMind fixed-concatenation Filter-Bank CSP experiment.

Each epoch is filtered into three predeclared bands, a two-component CSP is fitted
per band on the outer-training participants, and the six CSP features are
concatenated for the baseline LDA classifier. No feature selection or tuning is
performed. Evaluation is participant-held-out LOSO and uses a separate output
directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.metrics import confusion_matrix

import train_baseline_csp_lda as baseline
from run_frequency_band_comparison import filter_epochs

mne.set_log_level("WARNING")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = baseline.DEFAULT_DATASET
DEFAULT_OUTPUT = os.path.join(
    BASE_DIR, "output", "model_results", "fbcsp_fixed_concatenation_full_cohort"
)
DEFAULT_BASELINE_DIR = os.path.join(
    BASE_DIR, "output", "model_results", "baseline_reproduction_phase3"
)
CLASS_NAMES = {0: "Left_Fist", 1: "Right_Fist"}
BANDS = (
    ("mu_8_13_hz", "Mu", 8.0, 13.0),
    ("low_beta_13_20_hz", "Low Beta", 13.0, 20.0),
    ("high_beta_20_30_hz", "High Beta", 20.0, 30.0),
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob_right: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=baseline.CLASS_LABELS)
    metrics = baseline.metrics_from_cm(cm)
    metrics["roc_auc"] = baseline.safe_auc(y_true, prob_right)
    metrics["confusion_matrix"] = cm.astype(int).tolist()
    return metrics


def validate_predictions(
    predictions: pd.DataFrame,
    participant_ids: np.ndarray,
    y: np.ndarray,
) -> None:
    n_trials = len(y)
    expected_indices = np.arange(n_trials)
    if len(predictions) != n_trials or not np.array_equal(
        predictions["trial_index"].to_numpy(dtype=int), expected_indices
    ):
        raise RuntimeError("Expected exactly one held-out prediction per input trial.")

    indices = predictions["trial_index"].to_numpy(dtype=int)
    expected_participants = np.asarray(participant_ids).astype(str)[indices]
    actual_participants = predictions["participant_id"].astype(str).to_numpy()
    if not np.array_equal(actual_participants, expected_participants):
        raise RuntimeError("A prediction was assigned to the wrong held-out participant.")
    if not np.array_equal(predictions["true_label"].to_numpy(dtype=int), y):
        raise RuntimeError("Prediction labels do not match the input dataset.")


def run_loso(dataset: dict, band_data: dict[str, np.ndarray]):
    y = dataset["y"]
    participant_ids, run_ids = dataset["participant_ids"], dataset["run_ids"]
    participants = dataset["participants"]
    trial_rows = []
    participant_rows = []

    for fold, held_out in enumerate(participants, start=1):
        test_mask = participant_ids == held_out
        train_mask = ~test_mask
        y_train, y_test = y[train_mask], y[test_mask]
        if len(np.unique(y_train)) != 2:
            raise ValueError(
                f"Fold {fold} (held out {held_out}) lacks both classes in its training data."
            )
        if not np.any(test_mask):
            raise ValueError(f"Fold {fold}: held-out participant {held_out} has no trials.")

        train_features = []
        test_features = []
        try:
            for X_band in band_data.values():
                csp = baseline.make_csp()
                train_features.append(csp.fit_transform(X_band[train_mask], y_train))
                test_features.append(csp.transform(X_band[test_mask]))

            X_train_features = np.concatenate(train_features, axis=1)
            X_test_features = np.concatenate(test_features, axis=1)
            lda = baseline.make_lda()
            lda.fit(X_train_features, y_train)
            if list(lda.classes_) != baseline.CLASS_LABELS:
                raise RuntimeError(
                    f"Unexpected LDA class order {list(lda.classes_)}; "
                    f"expected {baseline.CLASS_LABELS}."
                )
            y_pred = lda.predict(X_test_features)
            probabilities = lda.predict_proba(X_test_features)
        except Exception as exc:
            raise RuntimeError(
                f"FBCSP fitting failed in fold {fold} (held out {held_out}): {exc}"
            ) from exc

        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(
                f"Non-finite probabilities in fold {fold} (held out {held_out})."
            )
        test_indices = np.flatnonzero(test_mask)
        for local_idx, trial_idx in enumerate(test_indices):
            trial_rows.append({
                "trial_index": int(trial_idx),
                "participant_id": str(held_out),
                "run_id": str(run_ids[trial_idx]),
                "true_label": int(y_test[local_idx]),
                "true_class_name": CLASS_NAMES[int(y_test[local_idx])],
                "predicted_label": int(y_pred[local_idx]),
                "predicted_class_name": CLASS_NAMES[int(y_pred[local_idx])],
                "prob_class0_left": float(probabilities[local_idx, 0]),
                "prob_class1_right": float(probabilities[local_idx, 1]),
                "is_correct": bool(y_pred[local_idx] == y_test[local_idx]),
                "loso_fold": int(fold),
            })

        fold_metrics = calculate_metrics(y_test, y_pred, probabilities[:, 1])
        participant_rows.append({
            "participant_id": str(held_out),
            "n_train_trials": int(np.sum(train_mask)),
            **fold_metrics,
        })
        print(f"Completed fold {fold}/{len(participants)}: held out {held_out}")

    predictions = (
        pd.DataFrame(trial_rows)
        .sort_values("trial_index")
        .reset_index(drop=True)
    )
    per_participant = pd.DataFrame(participant_rows)
    validate_predictions(predictions, participant_ids, y)
    return predictions, per_participant


def participant_metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant_id, group in predictions.groupby("participant_id", sort=True):
        rows.append({
            "participant_id": str(participant_id),
            **calculate_metrics(
                group["true_label"].to_numpy(dtype=int),
                group["predicted_label"].to_numpy(dtype=int),
                group["prob_class1_right"].to_numpy(dtype=float),
            ),
        })
    return pd.DataFrame(rows)


def load_and_validate_baseline(
    predictions_path: str,
    metrics_path: str,
    dataset: dict,
    dataset_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(f"Baseline predictions not found: {predictions_path}")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Baseline metrics JSON not found: {metrics_path}")

    with open(metrics_path, encoding="utf-8") as file:
        metrics_json = json.load(file)
    recorded_hash = metrics_json["run_environment"]["dataset_sha256"]
    if recorded_hash != dataset_hash:
        raise ValueError(
            "Baseline predictions were produced from a different dataset snapshot "
            f"(baseline SHA-256 {recorded_hash}; current SHA-256 {dataset_hash})."
        )
    model_configuration = metrics_json["model_configuration"]
    if (
        model_configuration["csp_parameters"] != baseline.CSP_PARAMS
        or model_configuration["lda_parameters"] != baseline.LDA_PARAMS
    ):
        raise ValueError(
            "The comparison predictions are not from the default CSP-2/LDA-SVD "
            "baseline configuration."
        )

    predictions = pd.read_csv(predictions_path).sort_values("trial_index").reset_index(drop=True)
    validate_predictions(predictions, dataset["participant_ids"], dataset["y"])
    per_participant = participant_metrics_from_predictions(predictions)
    expected = set(dataset["participants"])
    if set(per_participant["participant_id"]) != expected:
        raise ValueError("Baseline predictions do not cover the same participant cohort.")
    return predictions, per_participant


def compare_with_baseline(
    predictions: pd.DataFrame,
    per_participant: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_per_participant: pd.DataFrame,
) -> dict:
    y_true = predictions["true_label"].to_numpy(dtype=int)
    metrics = calculate_metrics(
        y_true,
        predictions["predicted_label"].to_numpy(dtype=int),
        predictions["prob_class1_right"].to_numpy(dtype=float),
    )
    baseline_metrics = calculate_metrics(
        baseline_predictions["true_label"].to_numpy(dtype=int),
        baseline_predictions["predicted_label"].to_numpy(dtype=int),
        baseline_predictions["prob_class1_right"].to_numpy(dtype=float),
    )
    participant_ids = sorted(per_participant["participant_id"].astype(str))
    fbcsp_indexed = per_participant.set_index("participant_id")
    baseline_indexed = baseline_per_participant.set_index("participant_id")
    paired = {}
    for metric in ("accuracy", "balanced_accuracy"):
        fbcsp_values = fbcsp_indexed.loc[participant_ids, metric].to_numpy(dtype=float)
        baseline_values = baseline_indexed.loc[participant_ids, metric].to_numpy(dtype=float)
        differences = fbcsp_values - baseline_values
        t_result = stats.ttest_rel(fbcsp_values, baseline_values)
        if np.allclose(differences, 0.0):
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
        else:
            w_result = stats.wilcoxon(differences)
            wilcoxon_stat, wilcoxon_p = float(w_result.statistic), float(w_result.pvalue)
        paired[metric] = {
            "mean_difference_fbcsp_minus_baseline": float(np.mean(differences)),
            "paired_t_statistic": float(t_result.statistic),
            "paired_t_p_value_two_sided": float(t_result.pvalue),
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_value_two_sided": wilcoxon_p,
            "n_participants": len(participant_ids),
        }

    return {
        "fbcsp": metrics,
        "baseline": baseline_metrics,
        "pooled_accuracy_difference_fbcsp_minus_baseline": (
            metrics["accuracy"] - baseline_metrics["accuracy"]
        ),
        "pooled_balanced_accuracy_difference_fbcsp_minus_baseline": (
            metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
        ),
        "paired_participant_tests": paired,
    }


def generate_figure(
    comparison: dict,
    per_participant: pd.DataFrame,
    baseline_per_participant: pd.DataFrame,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=250)
    names = ["Baseline CSP-LDA", "FBCSP"]
    accuracy = [
        comparison["baseline"]["accuracy"] * 100,
        comparison["fbcsp"]["accuracy"] * 100,
    ]
    balanced_accuracy = [
        comparison["baseline"]["balanced_accuracy"] * 100,
        comparison["fbcsp"]["balanced_accuracy"] * 100,
    ]
    x = np.arange(len(names))
    width = 0.34
    axes[0].bar(x - width / 2, accuracy, width, label="Accuracy", color="#2b5c8f")
    axes[0].bar(
        x + width / 2,
        balanced_accuracy,
        width,
        label="Balanced accuracy",
        color="#3498db",
    )
    axes[0].axhline(50, color="#7f8c8d", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Pooled held-out performance")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    baseline_indexed = baseline_per_participant.set_index("participant_id")
    fbcsp_indexed = per_participant.set_index("participant_id")
    participant_ids = sorted(fbcsp_indexed.index.astype(str))
    base_values = baseline_indexed.loc[participant_ids, "accuracy"].to_numpy() * 100
    fbcsp_values = fbcsp_indexed.loc[participant_ids, "accuracy"].to_numpy() * 100
    axes[1].scatter(base_values, fbcsp_values, color="#2b5c8f", alpha=0.75)
    axes[1].plot([0, 100], [0, 100], color="#7f8c8d", linestyle="--")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Baseline participant accuracy (%)")
    axes[1].set_ylabel("FBCSP participant accuracy (%)")
    axes[1].set_title("Paired participant comparison")
    axes[1].grid(linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def generate_report(
    dataset: dict,
    dataset_path: str,
    dataset_hash: str,
    baseline_path: str,
    output_path: str,
    comparison: dict,
    per_participant: pd.DataFrame,
    baseline_per_participant: pd.DataFrame,
) -> None:
    results = comparison["fbcsp"]
    base = comparison["baseline"]
    accuracy_test = comparison["paired_participant_tests"]["accuracy"]
    balanced_test = comparison["paired_participant_tests"]["balanced_accuracy"]
    participants = ", ".join(dataset["participants"])
    band_lines = "\n".join(
        f"- **{name}:** {low:g}–{high:g} Hz"
        for _, name, low, high in BANDS
    )
    baseline_per_mean = float(baseline_per_participant["accuracy"].mean())
    fbcsp_per_mean = float(per_participant["accuracy"].mean())
    per_participant_differences = (
        per_participant.set_index("participant_id")["accuracy"]
        - baseline_per_participant.set_index("participant_id")["accuracy"]
    ).to_numpy(dtype=float)
    participant_accuracy_difference = (fbcsp_per_mean - baseline_per_mean) * 100
    report = f"""# MotorMind — Fixed-Concatenation FBCSP (LOSO)

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Dataset:** `{os.path.abspath(dataset_path)}`
**Dataset SHA-256:** `{dataset_hash}`
**Cohort:** {len(dataset["participants"])} participants; {len(dataset["y"]):,} trials
**Participants:** {participants}
**Class counts:** Left fist={int(np.sum(dataset["y"] == 0)):,}; right fist={int(np.sum(dataset["y"] == 1)):,}
**Channels / sampling rate / epoch:** {", ".join(dataset["channel_names"])} / {dataset["sfreq"]:g} Hz / {dataset["epoch_duration_sec"]:g} s
**Baseline comparison:** `{os.path.abspath(baseline_path)}`

## Prespecified model

The model uses three frequency branches:
{band_lines}

Each branch fits a 2-component CSP on the outer-training participants only.
The six log-variance CSP features are concatenated and passed to the baseline
LDA (`solver="svd"`). This is the user's selected fixed-concatenation variant:
there is no feature selection or hyperparameter tuning. It is therefore a fixed
filter-bank CSP experiment rather than feature-selected canonical FBCSP.

The existing dataset is already band-pass filtered at 8–30 Hz. These additional
band filters are applied separately to the already-epoched 4-second trials with
MNE's zero-phase FIR (`firwin`, Hamming window, `reflect_limited` padding).
Filtering short epochs can introduce edge transients and is not equivalent to
filtering continuous recordings before epoching.

## Full-cohort results

| Metric | Baseline CSP-LDA | Fixed-concatenation FBCSP | Difference |
| :--- | ---: | ---: | ---: |
| Pooled accuracy | {base["accuracy"]:.2%} | {results["accuracy"]:.2%} | {comparison["pooled_accuracy_difference_fbcsp_minus_baseline"] * 100:+.2f} percentage points |
| Balanced accuracy | {base["balanced_accuracy"]:.2%} | {results["balanced_accuracy"]:.2%} | {comparison["pooled_balanced_accuracy_difference_fbcsp_minus_baseline"] * 100:+.2f} percentage points |
| Macro precision | {base["precision_macro"]:.2%} | {results["precision_macro"]:.2%} | — |
| Macro recall | {base["recall_macro"]:.2%} | {results["recall_macro"]:.2%} | — |
| Macro F1 | {base["f1_macro"]:.2%} | {results["f1_macro"]:.2%} | — |
| ROC AUC | {base["roc_auc"]:.4f} | {results["roc_auc"]:.4f} | — |
| Mean participant accuracy | {baseline_per_mean:.2%} | {fbcsp_per_mean:.2%} | {participant_accuracy_difference:+.2f} percentage points |

The FBCSP model achieved {int(np.sum(per_participant_differences > 1e-12))} higher,
{int(np.sum(per_participant_differences < -1e-12))} lower, and
{int(np.sum(np.isclose(per_participant_differences, 0.0, atol=1e-12)))} unchanged
participant accuracy scores relative to the baseline.

## Paired participant-level comparisons

| Measure | Mean difference (FBCSP − baseline) | Paired t-test p | Wilcoxon p |
| :--- | ---: | ---: | ---: |
| Accuracy | {accuracy_test["mean_difference_fbcsp_minus_baseline"] * 100:+.2f} percentage points | {accuracy_test["paired_t_p_value_two_sided"]:.4g} | {accuracy_test["wilcoxon_p_value_two_sided"]:.4g} |
| Balanced accuracy | {balanced_test["mean_difference_fbcsp_minus_baseline"] * 100:+.2f} percentage points | {balanced_test["paired_t_p_value_two_sided"]:.4g} | {balanced_test["wilcoxon_p_value_two_sided"]:.4g} |

The paired tests use participants as the unit of analysis. They are exploratory:
LOSO training sets overlap, and the p-values are not adjusted for multiple
comparisons. A higher point estimate alone is not evidence of a reliable
improvement.

## Validation and limitations

- Outer evaluation is strict participant-held-out LOSO: CSP and LDA are fitted
  using training participants; the held-out participant is used only for
  transformation and prediction.
- Exactly one prediction is recorded per trial in each held-out fold.
- The three band filters are fixed before evaluation; no outer-test score is used
  to select bands or features.
- The three-channel montage and short epochs limit how finely the frequency bank
  can be subdivided. The epoch-wise filtering caveat applies to all branches.
"""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-concatenation filter-bank CSP with participant-held-out LOSO."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Consolidated epochs NPZ.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="New output directory.")
    parser.add_argument(
        "--baseline-predictions",
        default=os.path.join(DEFAULT_BASELINE_DIR, "loso_predictions.csv"),
        help="Full-cohort baseline prediction CSV used for paired comparison.",
    )
    parser.add_argument(
        "--baseline-metrics",
        default=os.path.join(DEFAULT_BASELINE_DIR, "overall_metrics.json"),
        help="Baseline JSON containing its dataset SHA-256.",
    )
    args = parser.parse_args()
    dataset_path = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output)
    baseline_predictions_path = os.path.abspath(args.baseline_predictions)
    baseline_metrics_path = os.path.abspath(args.baseline_metrics)

    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if os.listdir(output_dir):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose a new --output path to preserve existing results."
            )

    dataset = baseline.load_and_validate_dataset(dataset_path)
    dataset_hash = baseline.sha256_of_file(dataset_path)
    baseline_predictions, baseline_per_participant = load_and_validate_baseline(
        baseline_predictions_path,
        baseline_metrics_path,
        dataset,
        dataset_hash,
    )
    print(f"Dataset SHA-256: {dataset_hash}")
    print(
        f"Evaluating {len(dataset['participants'])} LOSO folds "
        f"over {len(dataset['y']):,} trials."
    )

    band_data = {}
    for key, name, low, high in BANDS:
        band_data[key] = filter_epochs(
            dataset["X"],
            dataset["sfreq"],
            low,
            high,
            f"{low:g}–{high:g} Hz ({name})",
        )

    predictions, per_participant = run_loso(dataset, band_data)
    comparison = compare_with_baseline(
        predictions,
        per_participant,
        baseline_predictions,
        baseline_per_participant,
    )

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "fbcsp_loso_predictions.csv")
    participant_path = os.path.join(output_dir, "fbcsp_per_participant_metrics.csv")
    json_path = os.path.join(output_dir, "fbcsp_overall_metrics.json")
    figure_path = os.path.join(output_dir, "fbcsp_comparison.png")
    report_path = os.path.join(output_dir, "fbcsp_report.md")
    predictions.to_csv(predictions_path, index=False)
    per_participant.to_csv(participant_path, index=False)

    overall = {
        "experiment_name": "fbcsp_fixed_concatenation",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "path": dataset_path,
            "sha256": dataset_hash,
            "n_trials": len(dataset["y"]),
            "n_participants": len(dataset["participants"]),
            "participant_ids": dataset["participants"],
            "class_counts": {
                "left_fist": int(np.sum(dataset["y"] == 0)),
                "right_fist": int(np.sum(dataset["y"] == 1)),
            },
            "input_shape": list(dataset["X"].shape),
            "channels": dataset["channel_names"],
            "sampling_rate_hz": dataset["sfreq"],
            "epoch_duration_sec": dataset["epoch_duration_sec"],
            "units": dataset["units"],
        },
        "model": {
            "feature_extractor": "three-band filter-bank CSP",
            "bands_hz": {key: [low, high] for key, _, low, high in BANDS},
            "csp_parameters_per_band": baseline.CSP_PARAMS,
            "components_per_band": baseline.CSP_PARAMS["n_components"],
            "concatenated_feature_count": (
                len(BANDS) * baseline.CSP_PARAMS["n_components"]
            ),
            "feature_selection": "none; fixed concatenation",
            "classifier": "LinearDiscriminantAnalysis",
            "lda_parameters": baseline.LDA_PARAMS,
        },
        "evaluation": {
            "protocol": "Leave-One-Subject-Out",
            "n_folds": len(dataset["participants"]),
            "predictions_per_trial": 1,
            "filtering_caveat": (
                "Mu and Beta sub-bands are filtered separately on already-epoched "
                "data, not on continuous recordings."
            ),
        },
        "comparison_to_baseline": comparison,
        "baseline_artifacts": {
            "predictions_path": baseline_predictions_path,
            "metrics_path": baseline_metrics_path,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "mne": mne.__version__,
        },
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(baseline.to_jsonable(overall), file, indent=2, allow_nan=False)

    generate_figure(comparison, per_participant, baseline_per_participant, figure_path)
    generate_report(
        dataset,
        dataset_path,
        dataset_hash,
        baseline_predictions_path,
        report_path,
        comparison,
        per_participant,
        baseline_per_participant,
    )
    print(f"Saved FBCSP results to: {output_dir}")


if __name__ == "__main__":
    main()
