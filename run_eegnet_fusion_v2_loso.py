"""Run EEGNet Fusion V2 with strict participant-held-out LOSO validation.

The five-branch architecture follows Chowdhury et al. (2023). This runner keeps
MotorMind's existing 3-channel, 4-second epochs and reserves validation
participants from the outer training cohort; the outer test participant is
never used for training, model selection, or preprocessing.

Usage:
    python run_eegnet_fusion_v2_loso.py
    python run_eegnet_fusion_v2_loso.py --epochs 100 --device auto
    python run_eegnet_fusion_v2_loso.py --dataset PATH_TO.npz --output OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train_baseline_csp_lda as baseline

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_3CH_DATASET = baseline.DEFAULT_DATASET
DEFAULT_64CH_DATASET = os.path.join(
    BASE_DIR, "output", "epochs", "motormind_all_64ch_epochs.npz"
)
DEFAULT_DATASET = DEFAULT_3CH_DATASET
DEFAULT_3CH_OUTPUT = os.path.join(
    BASE_DIR, "output", "model_results", "eegnet_fusion_v2_3ch_loso"
)
DEFAULT_64CH_OUTPUT = os.path.join(
    BASE_DIR, "output", "model_results", "eegnet_fusion_v2_64ch_loso"
)
DEFAULT_OUTPUT = os.path.join(
    BASE_DIR, "output", "model_results", "eegnet_fusion_v2_loso"
)
DEFAULT_BASELINE_DIR = os.path.join(
    BASE_DIR, "output", "model_results", "baseline_reproduction_phase3"
)
CLASS_NAMES = {0: "Left_Fist", 1: "Right_Fist"}
CANONICAL_64_CHANNELS = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "FP1",
    "FPz", "FP2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1",
    "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7", "T8", "T9", "T10",
    "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
]
BRANCH_PARAMETERS = (
    (8, 16, 64, 8),
    (16, 32, 80, 16),
    (32, 64, 96, 32),
    (64, 128, 112, 64),
    (128, 256, 128, 128),
)


def load_and_validate_dataset(path: str) -> dict:
    """Load NPZ epochs (3-channel or 64-channel) with strict integrity checks."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        found = list(npz.files)
        missing = [k for k in baseline.REQUIRED_KEYS if k not in found]
        if missing:
            raise KeyError(f"NPZ is missing required keys {missing}. Keys found: {found}")
        X = np.asarray(npz["X"])
        y = np.asarray(npz["y"])
        participant_ids = np.asarray(npz["participant_ids"])
        run_ids = np.asarray(npz["run_ids"])
        channel_names = np.asarray(npz["channel_names"]).astype(str).ravel().tolist()
        sfreq = float(baseline._scalar(npz["sfreq"], "sfreq"))
        epoch_sec = float(baseline._scalar(npz["epoch_duration_sec"], "epoch_duration_sec"))
        units = str(baseline._scalar(npz["units"], "units")) if "units" in found else "not provided in NPZ"

    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (trials, channels, samples); got shape {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("X contains zero trials.")
    if X.shape[1] != len(channel_names):
        raise ValueError(f"X channel count {X.shape[1]} does not match channel_names length {len(channel_names)}.")
    if X.shape[2] != baseline.EXPECTED_N_SAMPLES:
        raise ValueError(f"X must have {baseline.EXPECTED_N_SAMPLES} samples; got {X.shape[2]}.")

    for name, arr in (("y", y), ("participant_ids", participant_ids), ("run_ids", run_ids)):
        if arr.ndim != 1:
            raise ValueError(f"'{name}' must be 1-D; got shape {arr.shape}.")
    n_trials = X.shape[0]
    if not (len(y) == len(participant_ids) == len(run_ids) == n_trials):
        raise ValueError(f"Length mismatch: X={n_trials}, y={len(y)}, "
                         f"participant_ids={len(participant_ids)}, run_ids={len(run_ids)}.")

    if len(channel_names) == 3:
        if channel_names != baseline.EXPECTED_CHANNELS:
            raise ValueError(f"3-channel dataset must have channels {baseline.EXPECTED_CHANNELS}; got {channel_names}.")
    elif len(channel_names) == 64:
        if channel_names != CANONICAL_64_CHANNELS:
            raise ValueError("64-channel dataset channel names mismatch canonical montage.")
    else:
        raise ValueError(f"Unsupported channel count: {len(channel_names)} (expected 3 or 64).")

    if not np.isclose(sfreq, baseline.EXPECTED_SFREQ_HZ):
        raise ValueError(f"Sampling frequency must be {baseline.EXPECTED_SFREQ_HZ} Hz; got {sfreq}.")
    if not np.isclose(epoch_sec, baseline.EXPECTED_EPOCH_SEC):
        raise ValueError(f"Epoch duration must be {baseline.EXPECTED_EPOCH_SEC} s; got {epoch_sec}.")

    if not np.issubdtype(y.dtype, np.number) and y.dtype != np.bool_:
        raise ValueError(f"y must be numeric 0/1 labels; got dtype {y.dtype}.")
    if not np.all(np.isfinite(y.astype(float))):
        raise ValueError("y contains NaN or infinite values.")
    unique_labels = np.unique(y)
    if not np.array_equal(unique_labels, np.array(baseline.CLASS_LABELS)):
        raise ValueError(f"Labels must be exactly {baseline.CLASS_LABELS}; found {unique_labels.tolist()}.")
    y = y.astype(int)

    finite = np.isfinite(X)
    if not finite.all():
        bad_trials = np.where(~finite.all(axis=(1, 2)))[0]
        raise ValueError(f"X contains NaN/inf in {len(bad_trials)} trials.")

    participants = sorted({str(pid) for pid in participant_ids})
    return {
        "X": X,
        "y": y,
        "participant_ids": participant_ids.astype(str),
        "run_ids": run_ids.astype(str),
        "channel_names": channel_names,
        "sfreq": sfreq,
        "epoch_duration_sec": epoch_sec,
        "units": units,
        "participants": participants,
    }



class EEGNetFusionBranch(nn.Module):
    """EEGNet-style temporal, depthwise-spatial, and separable-convolution branch."""

    def __init__(
        self,
        n_channels: int,
        f1: int,
        f2: int,
        temporal_kernel: int,
        separable_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        spatial_filters = f1 * 2
        self.depthwise_spatial = nn.Conv2d(
            f1,
            spatial_filters,
            kernel_size=(n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.depthwise_temporal = nn.Conv2d(
            spatial_filters,
            spatial_filters,
            kernel_size=(1, separable_kernel),
            padding="same",
            groups=spatial_filters,
            bias=False,
        )
        self.temporal = nn.Conv2d(
            1,
            f1,
            kernel_size=(1, temporal_kernel),
            padding="same",
            bias=False,
        )
        self.temporal_norm = nn.BatchNorm2d(f1, eps=1e-3, momentum=0.01)
        self.spatial_norm = nn.BatchNorm2d(
            spatial_filters, eps=1e-3, momentum=0.01
        )
        self.separable_pointwise = nn.Conv2d(
            spatial_filters, f2, kernel_size=1, bias=False
        )
        self.separable_norm = nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01)
        self.activation = nn.ELU()
        self.pool_temporal = nn.AvgPool2d(kernel_size=(1, 4))
        self.pool_features = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(dropout)

        self.n_channels = n_channels
        self.f1 = f1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.temporal_norm(x)
        x = self.depthwise_spatial(x)
        x = self.spatial_norm(x)
        x = self.activation(x)
        x = self.dropout(self.pool_temporal(x))
        x = self.depthwise_temporal(x)
        x = self.separable_pointwise(x)
        x = self.separable_norm(x)
        x = self.activation(x)
        x = self.dropout(self.pool_features(x))
        return torch.flatten(x, start_dim=1)

    @torch.no_grad()
    def apply_depthwise_max_norm(self, max_norm: float = 1.0) -> None:
        weights = self.depthwise_spatial.weight.view(self.f1 * 2, -1)
        norms = torch.linalg.vector_norm(weights, dim=1, keepdim=True).clamp_min(1e-12)
        weights.mul_(torch.clamp(max_norm / norms, max=1.0))


class EEGNetFusionV2(nn.Module):
    """Five-branch EEGNet Fusion V2 classifier for (trials, channels, samples)."""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if n_samples < 32:
            raise ValueError("EEGNet Fusion V2 requires at least 32 time samples.")
        self.branches = nn.ModuleList(
            EEGNetFusionBranch(n_channels, *parameters, dropout)
            for parameters in BRANCH_PARAMETERS
        )
        feature_samples = (n_samples // 4) // 8
        n_features = feature_samples * sum(parameters[1] for parameters in BRANCH_PARAMETERS)
        self.classifier = nn.Linear(n_features, len(baseline.CLASS_LABELS))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (trials, channels, samples); got {x.shape}.")
        x = x.unsqueeze(1)
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.classifier(features)

    @torch.no_grad()
    def apply_constraints(self) -> None:
        for branch in self.branches:
            branch.apply_depthwise_max_norm(1.0)
        self.classifier.weight.renorm_(p=2, dim=0, maxnorm=0.25)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_right: np.ndarray,
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=baseline.CLASS_LABELS)
    metrics = baseline.metrics_from_cm(cm)
    metrics["roc_auc"] = baseline.safe_auc(y_true, probability_right)
    metrics["confusion_matrix"] = cm.astype(int).tolist()
    return metrics


def validate_predictions(
    predictions: pd.DataFrame,
    participant_ids: np.ndarray,
    y: np.ndarray,
) -> None:
    expected_indices = np.arange(len(y))
    actual_indices = predictions["trial_index"].to_numpy(dtype=int)
    if len(predictions) != len(y) or not np.array_equal(actual_indices, expected_indices):
        raise RuntimeError("Expected exactly one held-out prediction per input trial.")
    expected_participants = participant_ids.astype(str)[actual_indices]
    if not np.array_equal(
        predictions["participant_id"].astype(str).to_numpy(), expected_participants
    ):
        raise RuntimeError("A prediction was assigned to the wrong held-out participant.")
    if not np.array_equal(predictions["true_label"].to_numpy(dtype=int), y):
        raise RuntimeError("Prediction labels do not match the input dataset.")


def validation_participants_for_fold(
    participants: list[str],
    held_out: str,
    fraction: float,
) -> list[str]:
    candidates = [
        participants[(participants.index(held_out) + offset) % len(participants)]
        for offset in range(1, len(participants))
    ]
    n_validation = max(1, int(math.ceil(len(candidates) * fraction)))
    return candidates[:n_validation]


def evaluate_model(
    model: EEGNetFusionV2,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    logits_batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            total_loss += float(loss_function(logits, labels).item())
            total_samples += len(labels)
            logits_batches.append(logits.cpu().numpy())
            labels_batches.append(labels.cpu().numpy())
    if total_samples == 0:
        raise ValueError("Validation or test data is empty.")
    return (
        total_loss / total_samples,
        np.concatenate(logits_batches),
        np.concatenate(labels_batches),
    )


def train_fold(
    dataset: dict,
    held_out: str,
    fold: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    participant_ids = dataset["participant_ids"]
    y = dataset["y"]
    test_mask = participant_ids == held_out
    validation_participants = validation_participants_for_fold(
        dataset["participants"], held_out, validation_fraction
    )
    validation_mask = np.isin(participant_ids, validation_participants)
    train_mask = ~(test_mask | validation_mask)

    y_train, y_validation, y_test = y[train_mask], y[validation_mask], y[test_mask]
    if len(np.unique(y_train)) != 2:
        raise ValueError(f"Fold {fold}: training participants do not contain both labels.")
    if not np.any(validation_mask) or not np.any(test_mask):
        raise ValueError(f"Fold {fold}: validation or held-out set is empty.")

    torch.manual_seed(seed + fold)
    random.seed(seed + fold)
    np.random.seed(seed + fold)

    x_all = dataset["X"].astype(np.float32, copy=False)
    train_data = TensorDataset(
        torch.from_numpy(x_all[train_mask]),
        torch.from_numpy(y_train.astype(np.int64, copy=False)),
    )
    validation_data = TensorDataset(
        torch.from_numpy(x_all[validation_mask]),
        torch.from_numpy(y_validation.astype(np.int64, copy=False)),
    )
    test_data = TensorDataset(
        torch.from_numpy(x_all[test_mask]),
        torch.from_numpy(y_test.astype(np.int64, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed + fold)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = EEGNetFusionV2(
        n_channels=x_all.shape[1],
        n_samples=x_all.shape[2],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    loss_function = nn.CrossEntropyLoss()

    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss in fold {fold}, epoch {epoch}."
                )
            loss.backward()
            optimizer.step()
            model.apply_constraints()

        validation_loss, _, _ = evaluate_model(model, validation_loader, device)
        if not np.isfinite(validation_loss):
            raise RuntimeError(
                f"Non-finite validation loss in fold {fold}, epoch {epoch}."
            )
        scheduler.step(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"Fold {fold} completed without a usable model checkpoint.")
    model.load_state_dict(best_state)
    _, logits, labels_test = evaluate_model(model, test_loader, device)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predicted = np.argmax(probabilities, axis=1)
    test_indices = np.flatnonzero(test_mask)
    rows = []
    for local_index, trial_index in enumerate(test_indices):
        true_label = int(labels_test[local_index])
        predicted_label = int(predicted[local_index])
        rows.append({
            "trial_index": int(trial_index),
            "participant_id": str(held_out),
            "run_id": str(dataset["run_ids"][trial_index]),
            "true_label": true_label,
            "true_class_name": CLASS_NAMES[true_label],
            "predicted_label": predicted_label,
            "predicted_class_name": CLASS_NAMES[predicted_label],
            "prob_class0_left": float(probabilities[local_index, 0]),
            "prob_class1_right": float(probabilities[local_index, 1]),
            "is_correct": bool(predicted_label == true_label),
            "loso_fold": fold,
        })

    fold_details = {
        "participant_id": held_out,
        "validation_participant_ids": validation_participants,
        "n_training_participants": int(len(np.unique(participant_ids[train_mask]))),
        "n_training_trials": int(np.sum(train_mask)),
        "n_validation_trials": int(np.sum(validation_mask)),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_seconds": time.perf_counter() - started,
    }
    return pd.DataFrame(rows), fold_details


def run_loso(
    dataset: dict,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    fold_rows = []
    participants = dataset["participants"]
    for fold, held_out in enumerate(participants, start=1):
        fold_predictions, fold_details = train_fold(
            dataset,
            held_out,
            fold,
            device,
            epochs,
            batch_size,
            learning_rate,
            validation_fraction,
            seed,
        )
        participant_metrics = calculate_metrics(
            fold_predictions["true_label"].to_numpy(dtype=int),
            fold_predictions["predicted_label"].to_numpy(dtype=int),
            fold_predictions["prob_class1_right"].to_numpy(dtype=float),
        )
        fold_rows.append({**fold_details, **participant_metrics})
        prediction_frames.append(fold_predictions)
        print(
            f"Completed fold {fold}/{len(participants)}: held out {held_out}; "
            f"accuracy {participant_metrics['accuracy']:.2%}; "
            f"best epoch {fold_details['best_epoch']}"
        )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values("trial_index")
        .reset_index(drop=True)
    )
    validate_predictions(predictions, dataset["participant_ids"], dataset["y"])
    return predictions, pd.DataFrame(fold_rows)


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
    model_configuration = metrics_json.get("model_configuration", {})
    if (
        model_configuration.get("csp_parameters") != baseline.CSP_PARAMS
        or model_configuration.get("lda_parameters") != baseline.LDA_PARAMS
    ):
        raise ValueError("Comparison baseline must use the default CSP-2/LDA-SVD configuration.")

    predictions = pd.read_csv(predictions_path).sort_values("trial_index").reset_index(drop=True)

    # Validate exact trial-by-trial alignment using participant, run, and trial identifiers
    if len(predictions) != len(dataset["y"]):
        raise ValueError(
            f"Baseline predictions row count ({len(predictions)}) does not match "
            f"dataset trial count ({len(dataset['y'])})."
        )
    if not np.array_equal(predictions["participant_id"].astype(str).to_numpy(), dataset["participant_ids"]):
        raise ValueError("Baseline predictions participant_ids do not match dataset.")
    if not np.array_equal(predictions["run_id"].astype(str).to_numpy(), dataset["run_ids"]):
        raise ValueError("Baseline predictions run_ids do not match dataset.")
    if not np.array_equal(predictions["true_label"].to_numpy(dtype=int), dataset["y"]):
        raise ValueError("Baseline predictions true labels do not match dataset.")

    per_participant = participant_metrics_from_predictions(predictions)
    if set(per_participant["participant_id"]) != set(dataset["participants"]):
        raise ValueError("Baseline predictions do not cover the same participant cohort.")

    baseline_hash = metrics_json.get("run_environment", {}).get("dataset_sha256", "")
    if baseline_hash == dataset_hash:
        print(f"Dataset SHA-256 matches baseline exactly: {dataset_hash[:12]}...")
    else:
        print(
            f"Baseline dataset SHA-256 ({baseline_hash[:12]}...) differs from current dataset "
            f"({dataset_hash[:12]}...); verified exact trial alignment across all {len(predictions):,} trials "
            "by (participant_id, run_id, trial_index, true_label)."
        )

    return predictions, per_participant


def compare_with_baseline(
    predictions: pd.DataFrame,
    per_participant: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_per_participant: pd.DataFrame,
) -> dict:
    metrics = calculate_metrics(
        predictions["true_label"].to_numpy(dtype=int),
        predictions["predicted_label"].to_numpy(dtype=int),
        predictions["prob_class1_right"].to_numpy(dtype=float),
    )
    baseline_metrics = calculate_metrics(
        baseline_predictions["true_label"].to_numpy(dtype=int),
        baseline_predictions["predicted_label"].to_numpy(dtype=int),
        baseline_predictions["prob_class1_right"].to_numpy(dtype=float),
    )
    ids = sorted(per_participant["participant_id"].astype(str))
    eegnet = per_participant.set_index("participant_id")
    baseline_indexed = baseline_per_participant.set_index("participant_id")
    paired = {}
    for metric in ("accuracy", "balanced_accuracy"):
        new_values = eegnet.loc[ids, metric].to_numpy(dtype=float)
        old_values = baseline_indexed.loc[ids, metric].to_numpy(dtype=float)
        differences = new_values - old_values
        t_test = stats.ttest_rel(new_values, old_values)
        if np.allclose(differences, 0.0):
            wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
        else:
            wilcoxon_result = stats.wilcoxon(differences)
            wilcoxon_statistic = float(wilcoxon_result.statistic)
            wilcoxon_p = float(wilcoxon_result.pvalue)
        paired[metric] = {
            "mean_difference": float(np.mean(differences)),
            "paired_t_statistic": float(t_test.statistic),
            "paired_t_p_value_two_sided": float(t_test.pvalue),
            "wilcoxon_statistic": wilcoxon_statistic,
            "wilcoxon_p_value_two_sided": wilcoxon_p,
            "n_participants": len(ids),
        }
    return {
        "eegnet": metrics,
        "baseline": baseline_metrics,
        "pooled_accuracy_difference": metrics["accuracy"] - baseline_metrics["accuracy"],
        "pooled_balanced_accuracy_difference": (
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
    ids = sorted(per_participant["participant_id"].astype(str))
    eegnet = per_participant.set_index("participant_id").loc[ids, "accuracy"].to_numpy() * 100
    old = baseline_per_participant.set_index("participant_id").loc[
        ids, "accuracy"
    ].to_numpy() * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
    names = ["CSP-LDA", "EEGNet Fusion V2"]
    axes[0].bar(
        names,
        [comparison["baseline"]["accuracy"] * 100, comparison["eegnet"]["accuracy"] * 100],
        color=["#3498db", "#9b59b6"],
    )
    axes[0].axhline(50, color="#7f8c8d", linestyle="--")
    axes[0].set_ylabel("Pooled held-out accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].scatter(old, eegnet, color="#6c3483", alpha=0.75)
    axes[1].plot([0, 100], [0, 100], color="#7f8c8d", linestyle="--")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Baseline accuracy per participant (%)")
    axes[1].set_ylabel("EEGNet accuracy per participant (%)")
    axes[1].grid(linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_report(
    output_path: str,
    dataset: dict,
    dataset_path: str,
    dataset_hash: str,
    baseline_path: str,
    comparison: dict,
    fold_metrics: pd.DataFrame,
    config: dict,
) -> None:
    eegnet = comparison["eegnet"]
    old = comparison["baseline"]
    accuracy_test = comparison["paired_participant_tests"]["accuracy"]
    report = f"""# EEGNet Fusion V2 — Strict Participant-Held-Out LOSO

**Run:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Dataset:** `{os.path.abspath(dataset_path)}`
**Dataset SHA-256:** `{dataset_hash}`
**Cohort:** {len(dataset["participants"])} participants; {len(dataset["y"]):,} trials
**Input:** {", ".join(dataset["channel_names"])}; {dataset["sfreq"]:g} Hz; {dataset["epoch_duration_sec"]:g}-second epochs
**Units:** {dataset["units"]}
**Baseline:** `{os.path.abspath(baseline_path)}`

## Model and training

This is a PyTorch implementation of the paper's five-branch EEGNet Fusion V2
architecture: temporal kernels 64/80/96/112/128, branch widths 8/16/32/64/128,
depth multiplier 2, separable convolution widths 8/16/32/64/128, average pooling
by 4 then 8, dropout 0.5, and a two-class fusion layer. It uses Adam at
{config["learning_rate"]:g}, batch size {config["batch_size"]}, and at most
{config["epochs"]} epochs. Best weights are selected by validation loss, with
ReduceLROnPlateau (factor 0.1, patience 5).

Each outer fold tests one participant only. From the remaining participants,
{config["validation_fraction"]:.0%} are reserved as participant-held-out
validation groups. The remaining groups train the model. The validation groups
are never part of training; the outer test participant is never used for model
selection. Each trial remains a single 4-second epoch; no trial windows are
split across partitions.

## Comparison

| Metric | CSP-LDA baseline | EEGNet Fusion V2 | Difference |
| --- | ---: | ---: | ---: |
| Pooled accuracy | {old["accuracy"]:.2%} | {eegnet["accuracy"]:.2%} | {comparison["pooled_accuracy_difference"] * 100:+.2f} percentage points |
| Balanced accuracy | {old["balanced_accuracy"]:.2%} | {eegnet["balanced_accuracy"]:.2%} | {comparison["pooled_balanced_accuracy_difference"] * 100:+.2f} percentage points |
| ROC AUC | {old["roc_auc"]:.4f} | {eegnet["roc_auc"]:.4f} | — |

Paired participant accuracy comparison: mean difference
{accuracy_test["mean_difference"] * 100:+.2f} percentage points;
paired t-test p={accuracy_test["paired_t_p_value_two_sided"]:.4g};
Wilcoxon p={accuracy_test["wilcoxon_p_value_two_sided"]:.4g}
(n={accuracy_test["n_participants"]} participants).

These paired tests are exploratory because LOSO training sets overlap.

## Paper-protocol differences and caveats

- Chowdhury et al. (2023) report random 70/10/20 sample splits, eight 80-sample
  windows per trial, and 64 EEG channels. MotorMind's evaluation intentionally
  differs: strict participant-held-out LOSO, the existing three-channel
  8–30 Hz dataset, and unsplit 4-second epochs. Its published scores are not
  directly comparable to this result.
- The paper's text lists Adam learning rate 0.00001, while its public
  implementation uses 0.001. This run follows the public implementation's
  0.001 setting unless overridden via `--learning-rate`.
- The 2023 article describes strict cross-subject performance, but its stated
  random sample split does not hold out participants. This run is the
  participant-held-out adaptation requested for MotorMind.

| Held-out participant | Accuracy | Balanced accuracy | Best epoch | Validation participants |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(f"| {row.participant_id} | {row.accuracy:.2%} | {row.balanced_accuracy:.2%} | {row.best_epoch} | {', '.join(row.validation_participant_ids)} |" for row in fold_metrics.itertuples())}
"""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EEGNet Fusion V2 with strict participant-held-out LOSO."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Consolidated epochs NPZ path (defaults to 3-channel dataset).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="New output directory (defaults to eegnet_fusion_v2_3ch_loso or eegnet_fusion_v2_64ch_loso).",
    )
    parser.add_argument(
        "--baseline-predictions",
        default=os.path.join(DEFAULT_BASELINE_DIR, "loso_predictions.csv"),
    )
    parser.add_argument(
        "--baseline-metrics",
        default=os.path.join(DEFAULT_BASELINE_DIR, "overall_metrics.json"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("--epochs, --batch-size, and --learning-rate must be positive.")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1.")
    if args.threads < 1:
        parser.error("--threads must be at least 1.")

    dataset_path = os.path.abspath(args.dataset)
    baseline_predictions_path = os.path.abspath(args.baseline_predictions)
    baseline_metrics_path = os.path.abspath(args.baseline_metrics)
    dataset = load_and_validate_dataset(dataset_path)
    dataset_hash = baseline.sha256_of_file(dataset_path)
    n_channels = dataset["X"].shape[1]

    if args.output is not None:
        output_dir = os.path.abspath(args.output)
    else:
        output_dir = DEFAULT_64CH_OUTPUT if n_channels == 64 else DEFAULT_3CH_OUTPUT

    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if os.listdir(output_dir):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose a new --output path to preserve existing results."
            )

    baseline_predictions, baseline_per_participant = load_and_validate_baseline(
        baseline_predictions_path,
        baseline_metrics_path,
        dataset,
        dataset_hash,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    torch.set_num_threads(args.threads)
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "device": str(device),
        "threads": args.threads,
    }
    print(
        f"Dataset SHA-256: {dataset_hash}\n"
        f"Running {len(dataset['participants'])} LOSO folds over "
        f"{len(dataset['y']):,} trials on {device}; up to {args.epochs} epochs/fold."
    )
    predictions, fold_metrics = run_loso(
        dataset,
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.validation_fraction,
        args.seed,
    )
    participant_metrics = participant_metrics_from_predictions(predictions)
    comparison = compare_with_baseline(
        predictions,
        participant_metrics,
        baseline_predictions,
        baseline_per_participant,
    )

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "eegnet_loso_predictions.csv")
    participant_path = os.path.join(output_dir, "eegnet_per_participant_metrics.csv")
    fold_path = os.path.join(output_dir, "eegnet_fold_training_metrics.csv")
    json_path = os.path.join(output_dir, "eegnet_overall_metrics.json")
    figure_path = os.path.join(output_dir, "eegnet_comparison.png")
    report_path = os.path.join(output_dir, "eegnet_report.md")
    predictions.to_csv(predictions_path, index=False)
    participant_metrics.to_csv(participant_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    generate_figure(comparison, participant_metrics, baseline_per_participant, figure_path)
    generate_report(
        report_path,
        dataset,
        dataset_path,
        dataset_hash,
        baseline_predictions_path,
        comparison,
        fold_metrics,
        config,
    )

    overall = {
        "experiment_name": "eegnet_fusion_v2_strict_loso",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "path": dataset_path,
            "sha256": dataset_hash,
            "n_trials": len(dataset["y"]),
            "n_participants": len(dataset["participants"]),
            "participant_ids": dataset["participants"],
            "input_shape": list(dataset["X"].shape),
            "channels": dataset["channel_names"],
            "sampling_rate_hz": dataset["sfreq"],
            "epoch_duration_sec": dataset["epoch_duration_sec"],
            "units": dataset["units"],
        },
        "model": {
            "name": "Five-branch EEGNet Fusion V2",
            "source_paper": (
                "Chowdhury et al., Sensors 2023, 23(18):7908, "
                "https://doi.org/10.3390/s23187908"
            ),
            "implementation_reference": (
                "https://github.com/radia-rayan-chowdhury/EEGNet-Fusion-V2"
            ),
            "branch_parameters_f1_f2_temporal_kernel_separable_kernel": [
                list(parameters) for parameters in BRANCH_PARAMETERS
            ],
            "depth_multiplier": 2,
            "dropout": 0.5,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "checkpoint_selection": "minimum participant-held-out validation cross-entropy",
            "lr_scheduler": "ReduceLROnPlateau(factor=0.1, patience=5)",
            "framework": f"PyTorch {torch.__version__}",
        },
        "evaluation": {
            "protocol": "outer participant-held-out Leave-One-Subject-Out",
            "n_folds": len(dataset["participants"]),
            "validation_split": (
                f"{args.validation_fraction:.0%} of outer-training participants, "
                "rotated deterministically by held-out participant"
            ),
            "predictions_per_trial": 1,
            "device": str(device),
            "seed": args.seed,
        },
        "comparison": comparison,
        "fold_training_metrics": fold_metrics.to_dict(orient="records"),
        "run_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "files": {
            "predictions": predictions_path,
            "per_participant_metrics": participant_path,
            "fold_training_metrics": fold_path,
            "figure": figure_path,
            "report": report_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(baseline.to_jsonable(overall), file, indent=2, allow_nan=False)
    print(
        f"\nPooled accuracy: {comparison['eegnet']['accuracy']:.2%} "
        f"(baseline {comparison['baseline']['accuracy']:.2%}; "
        f"difference {comparison['pooled_accuracy_difference'] * 100:+.2f} pp)"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
