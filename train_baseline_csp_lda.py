"""
MotorMind - Baseline experiment: CSP + LDA with Leave-One-Subject-Out (LOSO) validation
=======================================================================================

Task      : Binary classification of imagined left-fist (class 0) vs right-fist (class 1)
            movement from EEG channels C3, Cz, C4.
Features  : mne.decoding.CSP(n_components=2, reg=None, log=True, norm_trace=False,
            cov_est="concat", transform_into="average_power")
Classifier: sklearn LinearDiscriminantAnalysis(solver="svd")
Validation: Leave-One-Subject-Out. In every fold, CSP and LDA are fitted ONLY on the
            training participants; the held-out participant is only transformed/predicted.

This is a cross-subject (zero-calibration) generalization estimate. Nothing about the
held-out participant (signals, labels, statistics) is used for fitting, normalization,
feature selection or hyperparameter tuning.

All dataset sizes (trials, participants, folds, class counts) are computed from the
loaded NPZ file; nothing about the dataset size is hard-coded.

Outputs (default: <script folder>/output/model_results/baseline_csp_lda/):
    loso_predictions.csv, per_participant_metrics.csv, overall_metrics.json,
    confusion_matrix.png, training_report.md
Existing files with those names are copied to _backups/<timestamp>/ before overwrite.

Usage:
    python train_baseline_csp_lda.py
    python train_baseline_csp_lda.py --dataset PATH_TO.npz --output OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
import sklearn
from mne.decoding import CSP
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_auc_score

mne.set_log_level("WARNING")

# --------------------------------------------------------------------------------------
# Protocol constants (properties of the recording/preprocessing protocol, NOT dataset size)
# --------------------------------------------------------------------------------------
EXPECTED_CHANNELS = ["C3", "Cz", "C4"]
EXPECTED_SFREQ_HZ = 160.0
EXPECTED_EPOCH_SEC = 4.0
EXPECTED_N_SAMPLES = 640
CLASS_LABELS = [0, 1]
CLASS_NAMES = {0: "Left_Fist", 1: "Right_Fist"}
REQUIRED_KEYS = ["X", "y", "participant_ids", "run_ids", "channel_names", "sfreq", "epoch_duration_sec"]
OPTIONAL_KEYS = ["units"]

CSP_PARAMS = dict(n_components=2, reg=None, log=True, norm_trace=False,
                  cov_est="concat", transform_into="average_power")
LDA_PARAMS = dict(solver="svd")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(BASE_DIR, "output", "epochs", "motormind_all_participants_epochs.npz")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output", "model_results", "baseline_csp_lda")

OUTPUT_FILES = ["loso_predictions.csv", "per_participant_metrics.csv", "overall_metrics.json",
                "confusion_matrix.png", "training_report.md"]


# --------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------
def _scalar(arr, name):
    a = np.asarray(arr)
    if a.size != 1:
        raise ValueError(f"NPZ key '{name}' should be a single value, found shape {a.shape}.")
    return a.reshape(-1)[0]


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_dataset(path: str) -> dict:
    """Load the NPZ (read-only) and run all critical validations. Raises on any failure."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        found = list(npz.files)
        missing = [k for k in REQUIRED_KEYS if k not in found]
        if missing:
            raise KeyError(f"NPZ is missing required keys {missing}. Keys found: {found}")
        X = np.asarray(npz["X"])
        y = np.asarray(npz["y"])
        participant_ids = np.asarray(npz["participant_ids"])
        run_ids = np.asarray(npz["run_ids"])
        channel_names = np.asarray(npz["channel_names"]).astype(str).ravel().tolist()
        sfreq = float(_scalar(npz["sfreq"], "sfreq"))
        epoch_sec = float(_scalar(npz["epoch_duration_sec"], "epoch_duration_sec"))
        units = str(_scalar(npz["units"], "units")) if "units" in found else "not provided in NPZ"

    # Shape checks
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (trials, channels, samples); got shape {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("X contains zero trials.")
    if X.shape[1] != len(EXPECTED_CHANNELS) or X.shape[2] != EXPECTED_N_SAMPLES:
        raise ValueError(f"X must have shape (trials, {len(EXPECTED_CHANNELS)}, {EXPECTED_N_SAMPLES}); "
                         f"got {X.shape}.")
    for name, arr in (("y", y), ("participant_ids", participant_ids), ("run_ids", run_ids)):
        if arr.ndim != 1:
            raise ValueError(f"'{name}' must be 1-D; got shape {arr.shape}.")
    n_trials = X.shape[0]
    if not (len(y) == len(participant_ids) == len(run_ids) == n_trials):
        raise ValueError(f"Length mismatch: X={n_trials}, y={len(y)}, "
                         f"participant_ids={len(participant_ids)}, run_ids={len(run_ids)}.")

    # Metadata checks
    if channel_names != EXPECTED_CHANNELS:
        raise ValueError(f"Channel names must be exactly {EXPECTED_CHANNELS}; got {channel_names}.")
    if not np.isclose(sfreq, EXPECTED_SFREQ_HZ):
        raise ValueError(f"Sampling frequency must be {EXPECTED_SFREQ_HZ} Hz; got {sfreq}.")
    if not np.isclose(epoch_sec, EXPECTED_EPOCH_SEC):
        raise ValueError(f"Epoch duration must be {EXPECTED_EPOCH_SEC} s; got {epoch_sec}.")
    if int(round(sfreq * epoch_sec)) != X.shape[2]:
        raise ValueError(f"sfreq * epoch_duration ({sfreq * epoch_sec}) != samples per epoch ({X.shape[2]}).")

    # Labels
    if not np.issubdtype(y.dtype, np.number) and y.dtype != np.bool_:
        raise ValueError(f"y must be numeric 0/1 labels; got dtype {y.dtype}.")
    if not np.all(np.isfinite(y.astype(float))):
        raise ValueError("y contains NaN or infinite values.")
    unique_labels = np.unique(y)
    if not np.array_equal(unique_labels, np.array(CLASS_LABELS)):
        raise ValueError(f"Labels must be exactly {CLASS_LABELS} (both present); found {unique_labels.tolist()}.")
    y = y.astype(int)

    # EEG values
    finite = np.isfinite(X)
    if not finite.all():
        bad_trials = np.where(~finite.all(axis=(1, 2)))[0]
        raise ValueError(f"X contains {int((~finite).sum())} NaN/inf values in {len(bad_trials)} trials "
                         f"(first trial indices: {bad_trials[:10].tolist()}). Not silently skipping them.")
    X = X.astype(np.float64, copy=False)  # in-memory dtype cast for MNE; file is untouched

    # Participants
    participant_ids = participant_ids.astype(str)
    run_ids = run_ids.astype(str)
    participants = sorted(np.unique(participant_ids).tolist())
    if len(participants) < 2:
        raise ValueError(f"LOSO requires at least 2 participants; found {len(participants)}.")

    return {
        "X": X, "y": y, "participant_ids": participant_ids, "run_ids": run_ids,
        "participants": participants, "channel_names": channel_names, "sfreq": sfreq,
        "epoch_duration_sec": epoch_sec, "units": units, "npz_keys": found,
    }


# --------------------------------------------------------------------------------------
# Metrics (computed from confusion matrices; rows = true class, columns = predicted class)
# --------------------------------------------------------------------------------------
def _div(num, den):
    return float(num) / float(den) if den > 0 else float("nan")


def metrics_from_cm(cm: np.ndarray) -> dict:
    """cm is 2x2 with labels [0, 1]; rows=true, cols=predicted. Undefined values are NaN."""
    ll, lr = int(cm[0, 0]), int(cm[0, 1])  # true left  -> pred left / pred right
    rl, rr = int(cm[1, 0]), int(cm[1, 1])  # true right -> pred left / pred right
    n = ll + lr + rl + rr
    rec_l, rec_r = _div(ll, ll + lr), _div(rr, rl + rr)
    prec_l, prec_r = _div(ll, ll + rl), _div(rr, rr + lr)
    f1_l, f1_r = _div(2 * ll, 2 * ll + lr + rl), _div(2 * rr, 2 * rr + lr + rl)

    def macro(a, b):
        return float((a + b) / 2) if np.isfinite(a) and np.isfinite(b) else float("nan")

    return {
        "n_trials": n, "n_left": ll + lr, "n_right": rl + rr, "n_correct": ll + rr,
        "accuracy": _div(ll + rr, n),
        "balanced_accuracy": macro(rec_l, rec_r),
        "precision_macro": macro(prec_l, prec_r),
        "recall_macro": macro(rec_l, rec_r),
        "f1_macro": macro(f1_l, f1_r),
        "precision_left": prec_l, "recall_left": rec_l, "f1_left": f1_l,
        "precision_right": prec_r, "recall_right": rec_r, "f1_right": f1_r,
        "cm_left_as_left": ll, "cm_left_as_right": lr,
        "cm_right_as_left": rl, "cm_right_as_right": rr,
    }


def safe_auc(y_true, prob_class1):
    """ROC AUC from P(class 1). Returns NaN if undefined (only one class present)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, prob_class1))
    except ValueError:
        return float("nan")


def summarize(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(len(s))
    return {
        "n_defined": n,
        "mean": float(s.mean()) if n else float("nan"),
        "std_sample_ddof1": float(s.std(ddof=1)) if n > 1 else float("nan"),
        "median": float(s.median()) if n else float("nan"),
        "min": float(s.min()) if n else float("nan"),
        "max": float(s.max()) if n else float("nan"),
    }


# --------------------------------------------------------------------------------------
# LOSO evaluation
# --------------------------------------------------------------------------------------
def make_csp() -> CSP:
    return CSP(**CSP_PARAMS)


def make_lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(**LDA_PARAMS)


def run_loso(dataset: dict):
    X, y = dataset["X"], dataset["y"]
    pids, rids = dataset["participant_ids"], dataset["run_ids"]
    participants = dataset["participants"]
    n_folds = len(participants)

    trial_rows, participant_rows = [], []
    print("\nLOSO folds")
    print("-" * 92)
    print(f"{'Fold':<9}{'Held-out':<11}{'Train/Test':<14}{'Test L/R':<12}{'Acc':>8}{'BalAcc':>9}{'MacroF1':>9}{'AUC':>8}")
    print("-" * 92)

    for fold, held_out in enumerate(participants, start=1):
        test_mask = pids == held_out
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_idx = np.where(test_mask)[0]

        if len(np.unique(y_train)) != 2:
            raise ValueError(f"Fold {fold} (held out {held_out}): training set lacks one class "
                             f"(classes present: {np.unique(y_train).tolist()}).")
        if len(y_test) == 0:
            raise ValueError(f"Fold {fold}: held-out participant {held_out} has no trials.")
        if len(np.unique(y_test)) < 2:
            print(f"  WARNING: held-out participant {held_out} has only one class; "
                  f"balanced accuracy / AUC / some class metrics are undefined for this fold.")

        # --- Fit on training participants ONLY -------------------------------------
        try:
            csp = make_csp()
            F_train = csp.fit_transform(X_train, y_train)
            lda = make_lda()
            lda.fit(F_train, y_train)
        except Exception as exc:  # surface with fold context
            raise RuntimeError(f"Model fitting failed in fold {fold} (held out {held_out}): {exc}") from exc
        if list(lda.classes_) != CLASS_LABELS:
            raise RuntimeError(f"Unexpected LDA class order {list(lda.classes_)}; expected {CLASS_LABELS}.")

        # --- Held-out participant: transform + predict only ------------------------
        F_test = csp.transform(X_test)
        y_pred = lda.predict(F_test)
        proba = lda.predict_proba(F_test)
        if not np.all(np.isfinite(proba)):
            raise RuntimeError(f"Non-finite probabilities in fold {fold} (held out {held_out}).")

        for i, gi in enumerate(test_idx):
            trial_rows.append({
                "trial_index": int(gi),
                "participant_id": held_out,
                "run_id": str(rids[gi]),
                "true_label": int(y_test[i]),
                "true_class_name": CLASS_NAMES[int(y_test[i])],
                "predicted_label": int(y_pred[i]),
                "predicted_class_name": CLASS_NAMES[int(y_pred[i])],
                "prob_class0_left": float(proba[i, 0]),
                "prob_class1_right": float(proba[i, 1]),
                "is_correct": bool(y_pred[i] == y_test[i]),
                "loso_fold": fold,
            })

        cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
        m = metrics_from_cm(cm)
        auc = safe_auc(y_test, proba[:, 1])
        participant_rows.append({"participant_id": held_out, "n_train_trials": int(len(y_train)),
                                 **m, "roc_auc": auc})

        def f(v):
            return f"{v * 100:6.2f}%" if np.isfinite(v) else "   n/a "

        print(f"{fold:>3}/{n_folds:<5}{held_out:<11}{len(y_train):>5}/{len(y_test):<8}"
              f"{m['n_left']:>3}L/{m['n_right']:<4}R "
              f"{f(m['accuracy']):>8}{f(m['balanced_accuracy']):>9}{f(m['f1_macro']):>9}"
              f"{(f'{auc:.3f}' if np.isfinite(auc) else 'n/a'):>8}")
    print("-" * 92)

    df_pred = pd.DataFrame(trial_rows).sort_values("trial_index").reset_index(drop=True)
    df_part = pd.DataFrame(participant_rows)

    # Exactly one held-out prediction per trial
    n = len(X)
    if len(df_pred) != n or not np.array_equal(df_pred["trial_index"].to_numpy(), np.arange(n)):
        raise RuntimeError("Prediction bookkeeping error: not exactly one held-out prediction per trial.")
    return df_pred, df_part


def participant_level_test(df_part: pd.DataFrame) -> dict:
    """One-sided Wilcoxon signed-rank test of per-participant balanced accuracy against 0.5."""
    vals = df_part["balanced_accuracy"].dropna().to_numpy()
    diffs = vals - 0.5
    nonzero = diffs[diffs != 0]
    out = {
        "test": "one-sided Wilcoxon signed-rank test: per-participant balanced accuracy > 0.5",
        "unit_of_analysis": "participant (held-out fold)",
        "n_participants_with_defined_balanced_accuracy": int(len(vals)),
        "n_nonzero_differences": int(len(nonzero)),
        "alpha": 0.05,
        "statistic": None, "p_value": None, "performed": False,
        "caveats": ("Participants are treated as independent samples, but every fold's training set overlaps "
                    "with the others, so independence is approximate. The test says nothing about "
                    "individual participants and is not corrected for any other comparison."),
    }
    if len(nonzero) < 6:
        out["note"] = "Fewer than 6 non-zero differences; test not performed."
        return out
    res = stats.wilcoxon(nonzero, alternative="greater")
    out.update({"statistic": float(res.statistic), "p_value": float(res.pvalue), "performed": True})
    return out


def build_overall_metrics(dataset, df_pred, df_part, dataset_path, dataset_sha256) -> dict:
    X, y = dataset["X"], dataset["y"]
    participants = dataset["participants"]
    n_trials = int(len(y))
    n_left, n_right = int((y == 0).sum()), int((y == 1).sum())
    n_recordings = int(len({(p, r) for p, r in zip(dataset["participant_ids"], dataset["run_ids"])}))

    y_true = df_pred["true_label"].to_numpy()
    y_pred = df_pred["predicted_label"].to_numpy()
    p1 = df_pred["prob_class1_right"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    pooled = metrics_from_cm(cm)
    auc = safe_auc(y_true, p1)

    trials_per_part = df_part["n_trials"]
    return {
        "experiment_name": "baseline_csp_lda",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "mne": mne.__version__,
            "dataset_path": os.path.abspath(dataset_path), "dataset_sha256": dataset_sha256,
        },
        "dataset_summary": {
            "n_trials": n_trials,
            "n_participants": len(participants),
            "participant_ids": participants,
            "n_recordings_participant_run_pairs": n_recordings,
            "trials_per_participant": {"min": int(trials_per_part.min()), "max": int(trials_per_part.max()),
                                       "mean": float(trials_per_part.mean())},
            "input_shape": list(X.shape),
            "channels": dataset["channel_names"],
            "sampling_frequency_hz": dataset["sfreq"],
            "epoch_duration_sec": dataset["epoch_duration_sec"],
            "samples_per_epoch": int(X.shape[2]),
            "units_as_stored": dataset["units"],
            "class_distribution": {"0_Left_Fist": n_left, "1_Right_Fist": n_right},
            "class_proportions": {"0_Left_Fist": n_left / n_trials, "1_Right_Fist": n_right / n_trials},
            "majority_class_accuracy": max(n_left, n_right) / n_trials,
            "preprocessing_as_documented_by_project": "8-30 Hz bandpass (not verified by this script)",
        },
        "model_configuration": {
            "feature_extractor": "mne.decoding.CSP", "csp_parameters": CSP_PARAMS,
            "classifier": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "lda_parameters": LDA_PARAMS,
            "hyperparameter_tuning": "none",
        },
        "evaluation_protocol": {
            "method": "Leave-One-Subject-Out (LOSO) cross-validation",
            "interpretation": "Cross-subject generalization to held-out participants (zero calibration).",
            "n_folds": len(participants),
            "per_fold_procedure": [
                "hold out all trials of one participant",
                "fit CSP on training participants only",
                "fit LDA on training participants' CSP features only",
                "transform and predict the held-out participant with the fitted models",
            ],
            "predictions_per_trial": 1,
            "confusion_matrix_convention": "rows = true class, columns = predicted class, order [0 Left_Fist, 1 Right_Fist]",
            "roc_auc_source": "predicted probability of class 1 (Right_Fist), not predicted labels",
            "undefined_metric_handling": "reported as null/NaN; never replaced by 0",
        },
        "pooled_heldout_metrics": {
            "n_heldout_predictions": int(len(df_pred)),
            "accuracy": pooled["accuracy"],
            "balanced_accuracy": pooled["balanced_accuracy"],
            "roc_auc": auc,
            "macro_precision": pooled["precision_macro"],
            "macro_recall": pooled["recall_macro"],
            "macro_f1": pooled["f1_macro"],
            "class_0_left_fist": {"support": pooled["n_left"], "precision": pooled["precision_left"],
                                  "recall": pooled["recall_left"], "f1": pooled["f1_left"]},
            "class_1_right_fist": {"support": pooled["n_right"], "precision": pooled["precision_right"],
                                   "recall": pooled["recall_right"], "f1": pooled["f1_right"]},
            "confusion_matrix": {
                "labels": ["Left_Fist(0)", "Right_Fist(1)"],
                "matrix_true_rows_pred_cols": cm.astype(int).tolist(),
            },
        },
        "participant_level_summary": {
            "note": "Each participant is one unit; std is the sample standard deviation (ddof=1).",
            "accuracy": summarize(df_part["accuracy"]),
            "balanced_accuracy": summarize(df_part["balanced_accuracy"]),
            "macro_f1": summarize(df_part["f1_macro"]),
            "recall_left": summarize(df_part["recall_left"]),
            "recall_right": summarize(df_part["recall_right"]),
            "roc_auc": summarize(df_part["roc_auc"]),
        },
        "statistical_assessment": participant_level_test(df_part),
        "notes": [
            "Pooled accuracy weights participants by trial count; mean participant accuracy weights them equally.",
            "Trials were validated structurally only (shape, finite values, labels, metadata); signal quality "
            "and artifacts were not assessed and no trials were excluded.",
        ],
    }


# --------------------------------------------------------------------------------------
# Output writing
# --------------------------------------------------------------------------------------
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def backup_existing_outputs(output_dir: str):
    existing = [f for f in OUTPUT_FILES if os.path.exists(os.path.join(output_dir, f))]
    if not existing:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(output_dir, "_backups", stamp)
    os.makedirs(backup_dir, exist_ok=True)
    for f in existing:
        shutil.copy2(os.path.join(output_dir, f), os.path.join(backup_dir, f))
    return backup_dir


def pct(v, digits=2):
    return f"{v * 100:.{digits}f}%" if v is not None and np.isfinite(v) else "n/a"


def num(v, digits=4):
    return f"{v:.{digits}f}" if v is not None and np.isfinite(v) else "n/a"


def generate_figure(df_part: pd.DataFrame, overall: dict, path: str):
    pm = overall["pooled_heldout_metrics"]
    ds = overall["dataset_summary"]
    cm = np.array(pm["confusion_matrix"]["matrix_true_rows_pred_cols"])
    sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, sums, out=np.zeros(cm.shape, dtype=float), where=sums > 0)

    n_part = len(df_part)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, 7 + 0.3 * n_part), 6.5), dpi=200,
                             gridspec_kw={"width_ratios": [1, max(1.4, n_part / 14)]})

    ax = axes[0]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Row-normalised (recall)")
    labels = ["Left_Fist\n(0)", "Right_Fist\n(1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"A. Pooled held-out confusion matrix\nN = {ds['n_trials']} trials, "
                 f"{overall['evaluation_protocol']['n_folds']} LOSO folds\n"
                 f"Pooled accuracy = {pct(pm['accuracy'], 1)}", fontsize=11)

    ax = axes[1]
    ids = df_part["participant_id"].tolist()
    acc = df_part["accuracy"].to_numpy() * 100
    colors = ["#2b5c8f" if a >= 50 else "#c0392b" for a in acc]
    bars = ax.bar(range(n_part), acc, color=colors, edgecolor="black", linewidth=0.6)
    mean_acc = overall["participant_level_summary"]["accuracy"]["mean"] * 100
    ax.axhline(50, color="#7f8c8d", linestyle="--", linewidth=1.3, label="50% reference line")
    ax.axhline(mean_acc, color="#e67e22", linewidth=1.8, label=f"Mean of participant accuracies ({mean_acc:.1f}%)")
    ax.axhline(pm["accuracy"] * 100, color="#27ae60", linestyle=":", linewidth=1.8,
               label=f"Pooled accuracy ({pm['accuracy'] * 100:.1f}%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(range(n_part)); ax.set_xticklabels(ids, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy on held-out participant (%)"); ax.set_xlabel("Held-out participant")
    ax.set_title(f"B. Accuracy per held-out participant (n = {n_part})\nblue >= 50%, red < 50%", fontsize=11)
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="y", linestyle=":", alpha=0.6)
    if n_part <= 40:
        for b, a in zip(bars, acc):
            ax.annotate(f"{a:.0f}", (b.get_x() + b.get_width() / 2, a), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def generate_report(overall: dict, df_part: pd.DataFrame, path: str, dataset_path: str, command: str):
    ds, pm = overall["dataset_summary"], overall["pooled_heldout_metrics"]
    pl, st = overall["participant_level_summary"], overall["statistical_assessment"]
    env = overall["run_environment"]
    cm = pm["confusion_matrix"]["matrix_true_rows_pred_cols"]
    n_part = ds["n_participants"]
    ids = ds["participant_ids"]
    cd = ds["class_distribution"]

    best = df_part.loc[df_part["accuracy"].idxmax()]
    worst = df_part.loc[df_part["accuracy"].idxmin()]
    n_above = int((df_part["accuracy"] > 0.5).sum())
    n_at = int((df_part["accuracy"] == 0.5).sum())
    n_below = int((df_part["accuracy"] < 0.5).sum())

    L = []
    L.append("# MotorMind - Baseline Report: CSP + LDA with Leave-One-Subject-Out Validation\n")
    L.append(f"**Run timestamp:** {overall['timestamp']}  ")
    L.append(f"**Dataset:** `{os.path.abspath(dataset_path)}` (SHA-256 `{env['dataset_sha256']}`)  ")
    L.append("**Validation type:** cross-subject generalization on held-out participants (no per-subject calibration)\n")
    L.append("*This report was generated automatically from the results of this run.*\n")

    L.append("## 1. Summary\n")
    L.append(f"The dataset contains **{ds['n_trials']} trials** from **{n_part} participants** "
             f"({ids[0]} to {ids[-1]}, {ds['n_recordings_participant_run_pairs']} participant-run recordings). "
             f"Class counts: Left_Fist = {cd['0_Left_Fist']} ({pct(ds['class_proportions']['0_Left_Fist'])}), "
             f"Right_Fist = {cd['1_Right_Fist']} ({pct(ds['class_proportions']['1_Right_Fist'])}). "
             f"A majority-class predictor would reach {pct(ds['majority_class_accuracy'])} accuracy.\n")
    L.append(f"- **Pooled accuracy** ({pm['n_heldout_predictions']} held-out predictions): **{pct(pm['accuracy'])}**")
    L.append(f"- **Pooled balanced accuracy:** {pct(pm['balanced_accuracy'])}")
    L.append(f"- **Pooled ROC AUC** (from P(Right_Fist)): {num(pm['roc_auc'])}")
    L.append(f"- **Pooled macro F1:** {pct(pm['macro_f1'])}")
    L.append(f"- **Mean participant accuracy:** {pct(pl['accuracy']['mean'])} "
             f"(sample SD {pct(pl['accuracy']['std_sample_ddof1'])}, median {pct(pl['accuracy']['median'])}, "
             f"range {pct(pl['accuracy']['min'])} to {pct(pl['accuracy']['max'])}, n = {pl['accuracy']['n_defined']})")
    L.append(f"- **Mean participant balanced accuracy:** {pct(pl['balanced_accuracy']['mean'])} "
             f"(SD {pct(pl['balanced_accuracy']['std_sample_ddof1'])}, n = {pl['balanced_accuracy']['n_defined']})\n")
    L.append("Pooled accuracy weights each participant by their number of trials, whereas the mean participant "
             "accuracy weights every participant equally; the two can differ.\n")

    L.append("## 2. Data and protocol\n")
    L.append(f"- Input shape: {tuple(ds['input_shape'])} (trials, channels, samples)")
    L.append(f"- Channels: {', '.join(ds['channels'])}")
    L.append(f"- Sampling frequency: {ds['sampling_frequency_hz']:g} Hz; epoch duration: {ds['epoch_duration_sec']:g} s "
             f"({ds['samples_per_epoch']} samples)")
    L.append(f"- Units stored in NPZ: {ds['units_as_stored']}")
    L.append(f"- Preprocessing (as documented by the project, not re-verified here): {ds['preprocessing_as_documented_by_project']}")
    L.append(f"- Trials per participant: min {ds['trials_per_participant']['min']}, "
             f"max {ds['trials_per_participant']['max']}, mean {ds['trials_per_participant']['mean']:.1f}\n")
    L.append("### Validation procedure\n")
    L.append(f"Leave-One-Subject-Out cross-validation with {overall['evaluation_protocol']['n_folds']} folds. "
             "In each fold all trials of one participant are held out; CSP and then LDA are fitted only on the "
             "remaining participants; the held-out trials are transformed and classified with the fitted models. "
             "Each trial receives exactly one held-out prediction. Held-out labels and signals are not used for "
             "fitting, normalization, feature selection or hyperparameter tuning. This estimates generalization "
             "to **unseen participants**; it does not estimate within-participant (calibrated) performance.\n")

    L.append("## 3. Model\n")
    L.append(f"- Features: `mne.decoding.CSP` with {overall['model_configuration']['csp_parameters']}")
    L.append(f"- Classifier: `LinearDiscriminantAnalysis` with {overall['model_configuration']['lda_parameters']}")
    L.append("- No hyperparameters were tuned.\n")

    L.append("## 4. Pooled held-out results\n")
    L.append("Confusion matrix (rows = true class, columns = predicted class):\n")
    L.append("| | Pred Left_Fist (0) | Pred Right_Fist (1) |")
    L.append("| :--- | ---: | ---: |")
    L.append(f"| **True Left_Fist (0)** | {cm[0][0]} | {cm[0][1]} |")
    L.append(f"| **True Right_Fist (1)** | {cm[1][0]} | {cm[1][1]} |\n")
    c0, c1 = pm["class_0_left_fist"], pm["class_1_right_fist"]
    L.append("| Class | Support | Precision | Recall | F1 |")
    L.append("| :--- | ---: | ---: | ---: | ---: |")
    L.append(f"| Left_Fist (0) | {c0['support']} | {pct(c0['precision'])} | {pct(c0['recall'])} | {pct(c0['f1'])} |")
    L.append(f"| Right_Fist (1) | {c1['support']} | {pct(c1['precision'])} | {pct(c1['recall'])} | {pct(c1['f1'])} |\n")
    L.append("Reference levels: balanced accuracy and ROC AUC of an uninformative binary classifier are 0.5 in "
             f"expectation. Plain accuracy of an uninformative classifier depends on class balance (majority-class "
             f"rate here: {pct(ds['majority_class_accuracy'])}). Macro F1 has no single universal chance level, "
             "because it depends on the class balance and on the classifier's prediction rates.\n")

    L.append("## 5. Per-participant results\n")
    L.append("| Participant | Trials | L / R | Accuracy | Balanced acc. | Macro F1 | AUC | L->L | L->R | R->L | R->R |")
    L.append("| :---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in df_part.iterrows():
        L.append(f"| `{r['participant_id']}` | {int(r['n_trials'])} | {int(r['n_left'])} / {int(r['n_right'])} | "
                 f"{pct(r['accuracy'])} | {pct(r['balanced_accuracy'])} | {pct(r['f1_macro'])} | {num(r['roc_auc'], 3)} | "
                 f"{int(r['cm_left_as_left'])} | {int(r['cm_left_as_right'])} | "
                 f"{int(r['cm_right_as_left'])} | {int(r['cm_right_as_right'])} |")
    L.append("\n(L->R = true Left_Fist predicted as Right_Fist, etc. 'n/a' marks an undefined value.)\n")
    L.append(f"Participants with accuracy above / equal to / below 50%: **{n_above} / {n_at} / {n_below}** of {n_part}. "
             f"Highest accuracy: `{best['participant_id']}` ({pct(best['accuracy'])}); "
             f"lowest: `{worst['participant_id']}` ({pct(worst['accuracy'])}). "
             "Because each participant contributes a limited number of trials, individual accuracies are noisy "
             "and a value near 50% is not by itself evidence of non-responsiveness.\n")
    L.append("Class-specific recall across participants: "
             f"Left_Fist mean {pct(pl['recall_left']['mean'])} (SD {pct(pl['recall_left']['std_sample_ddof1'])}), "
             f"Right_Fist mean {pct(pl['recall_right']['mean'])} (SD {pct(pl['recall_right']['std_sample_ddof1'])}).\n")

    L.append("## 6. Statistical assessment\n")
    if st["performed"]:
        verdict = ("the test rejected the null hypothesis that participant balanced accuracy is centred at or below 0.5"
                   if st["p_value"] < st["alpha"] else
                   "the test did not reject the null hypothesis that participant balanced accuracy is centred at or below 0.5")
        L.append(f"A {st['test']} was performed over n = {st['n_nonzero_differences']} participants with non-zero "
                 f"difference from 0.5 (W = {st['statistic']:.2f}, p = {st['p_value']:.4g}, alpha = {st['alpha']}); "
                 f"{verdict}. {st['caveats']}\n")
    else:
        L.append(f"No significance test was performed: {st.get('note', 'insufficient data')} "
                 "No claim of statistical significance is made.\n")
    L.append("No test was run on the pooled trial-level accuracy, because trials from one participant are not "
             "independent and a pooled binomial test would overstate certainty.\n")

    L.append("## 7. Data-quality statement\n")
    L.append("The script verified array shapes, matching lengths, channel names, sampling frequency, epoch duration, "
             "label values and that all EEG values are finite. It did **not** assess artifacts, bad channels, "
             "amplitude outliers or inter-participant amplitude differences, and it excluded no trials. The presence "
             "of a trial in the NPZ file therefore does not imply that it is artifact-free or otherwise valid.\n")

    L.append("## 8. Limitations\n")
    L.append("- Offline analysis of fixed 4 s epochs; not an asynchronous or real-time BCI evaluation.")
    L.append("- Only 3 channels (C3, Cz, C4), so CSP has at most 3 spatial filters to choose from; 2 are used.")
    L.append("- Standard CSP with no regularization or per-participant normalization; amplitude differences between "
             "participants may influence the covariance estimates.")
    L.append("- Results measure generalization across participants in this dataset only and may not transfer to other "
             "recording setups or populations.")
    L.append("- This is a research baseline, not a clinically validated system.\n")

    L.append("## 9. Reproducibility\n")
    L.append(f"- Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, scipy {env['scipy']}, "
             f"scikit-learn {env['scikit_learn']}, mne {env['mne']}")
    L.append("- No random components are used in CSP/LDA (svd solver) or in the folds.")
    L.append(f"- Command: `{command}`\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="MotorMind baseline: CSP + LDA, LOSO validation")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to consolidated epochs NPZ")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    args = ap.parse_args()

    print("=" * 92)
    print("MotorMind baseline: CSP + LDA, Leave-One-Subject-Out cross-validation")
    print("=" * 92)

    dataset = load_and_validate_dataset(args.dataset)
    X, y = dataset["X"], dataset["y"]
    parts = dataset["participants"]
    print(f"Dataset        : {args.dataset}")
    print(f"Trials         : {len(X)}  (Left={int((y == 0).sum())}, Right={int((y == 1).sum())})")
    print(f"Participants   : {len(parts)}  ({parts[0]} .. {parts[-1]})")
    print(f"Epoch shape    : {X.shape[1]} channels x {X.shape[2]} samples @ {dataset['sfreq']:g} Hz")
    print(f"LOSO folds     : {len(parts)}")
    print("Validation checks passed.")

    dataset_sha = sha256_of_file(args.dataset)
    df_pred, df_part = run_loso(dataset)
    overall = build_overall_metrics(dataset, df_pred, df_part, args.dataset, dataset_sha)

    pm = overall["pooled_heldout_metrics"]
    pl = overall["participant_level_summary"]
    print("\nPooled held-out results")
    print(f"  Predictions        : {pm['n_heldout_predictions']}")
    print(f"  Pooled accuracy    : {pct(pm['accuracy'])}")
    print(f"  Balanced accuracy  : {pct(pm['balanced_accuracy'])}")
    print(f"  Macro F1           : {pct(pm['macro_f1'])}")
    print(f"  ROC AUC            : {num(pm['roc_auc'])}")
    print(f"  Participant mean acc: {pct(pl['accuracy']['mean'])} +/- {pct(pl['accuracy']['std_sample_ddof1'])} (SD)")

    os.makedirs(args.output, exist_ok=True)
    backup_dir = backup_existing_outputs(args.output)
    if backup_dir:
        print(f"\nExisting outputs backed up to: {backup_dir}")

    df_pred.to_csv(os.path.join(args.output, "loso_predictions.csv"), index=False)
    df_part.to_csv(os.path.join(args.output, "per_participant_metrics.csv"), index=False)
    with open(os.path.join(args.output, "overall_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(overall), f, indent=2, allow_nan=False)
    generate_figure(df_part, overall, os.path.join(args.output, "confusion_matrix.png"))
    command = "python " + " ".join(os.path.basename(a) if i == 0 else a for i, a in enumerate(sys.argv))
    generate_report(overall, df_part, os.path.join(args.output, "training_report.md"), args.dataset, command)

    print(f"\nSaved 5 output files to: {args.output}")


if __name__ == "__main__":
    main()