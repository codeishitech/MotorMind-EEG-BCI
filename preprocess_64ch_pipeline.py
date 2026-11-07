"""
MotorMind — 64-Channel Preprocessing Pipeline (Experiment A)
============================================================
Processes PhysioNet EEGMMIDB motor-imagery runs R04, R08, R12 across
participants S001 through S070, preserving all 64 sanitized EEG channels.

Uses the verified preprocessing methodology from preprocessing_pipeline.py:
  - 10-10 channel name sanitization
  - Strict EDF header validation
  - Raw zero-signal interval scanning
  - Continuous zero-phase windowed sinc FIR bandpass (8.0–30.0 Hz)
  - 4.0-second epoch extraction (640 samples @ 160 Hz)
  - Trial quality screening (zero collision, finite values, flatline detection)

Outputs (strictly to new paths):
  - output/epochs/motormind_all_64ch_epochs.npz
  - output/epochs/motormind_trial_metadata_64ch.csv
  - output/dataset_manifest_64ch.csv
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import mne
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing_pipeline import (
    SAMPLING_FREQ_HZ,
    EPOCH_DURATION_SEC,
    PASSBAND_LOW_HZ,
    PASSBAND_HIGH_HZ,
    LABEL_LEFT_FIST,
    LABEL_RIGHT_FIST,
    LABEL_NAMES,
    sanitize_channel_names,
    validate_edf_recording,
    detect_zero_intervals,
    filter_eeg_data,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
EPOCHS_DIR = os.path.join(OUTPUT_DIR, "epochs")

TARGET_NPZ_PATH = os.path.join(EPOCHS_DIR, "motormind_all_64ch_epochs.npz")
TARGET_TRIAL_META_PATH = os.path.join(EPOCHS_DIR, "motormind_trial_metadata_64ch.csv")
TARGET_MANIFEST_PATH = os.path.join(OUTPUT_DIR, "dataset_manifest_64ch.csv")

PARTICIPANTS = [f"S{i:03d}" for i in range(1, 71)]
RUNS = ["R04", "R08", "R12"]


def extract_64ch_epochs(
    raw_filt: mne.io.BaseRaw,
    zero_intervals: list[tuple[float, float, float, int]],
    participant_id: str,
    run_id: str,
    channel_names: list[str],
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    """Extracts 4.0-second epochs (640 samples @ 160 Hz) across all 64 channels."""
    sfreq = float(raw_filt.info["sfreq"])
    events, event_id = mne.events_from_annotations(raw_filt, verbose=False)

    t1_code = event_id.get("T1", -1)
    t2_code = event_id.get("T2", -1)
    task_events = [ev for ev in events if ev[2] in [t1_code, t2_code]]

    retained_epochs: list[np.ndarray] = []
    retained_labels: list[int] = []
    trial_audit_rows: list[dict] = []

    filt_data = raw_filt.get_data()  # shape: (64, n_times) in Volts

    # Identify C3, Cz, C4 indices for metadata logging
    c3_idx = channel_names.index("C3") if "C3" in channel_names else None
    cz_idx = channel_names.index("Cz") if "Cz" in channel_names else None
    c4_idx = channel_names.index("C4") if "C4" in channel_names else None

    for tr_idx, ev in enumerate(task_events):
        onset_sample = int(ev[0])
        onset_sec = onset_sample / sfreq
        duration_sec = EPOCH_DURATION_SEC
        end_sec = onset_sec + duration_sec
        n_samples = int(duration_sec * sfreq)
        end_sample = onset_sample + n_samples

        cue_code = "T1" if ev[2] == t1_code else "T2"
        class_label = LABEL_LEFT_FIST if cue_code == "T1" else LABEL_RIGHT_FIST
        label_name = LABEL_NAMES[class_label]

        # Check zero-dropout overlap
        has_zero_overlap = False
        overlap_details = ""
        for zs, ze, zd, zcount in zero_intervals:
            if max(onset_sec, zs) < min(end_sec, ze):
                has_zero_overlap = True
                overlap_sec = min(end_sec, ze) - max(onset_sec, zs)
                overlap_details = f"Overlaps zero interval [{zs:.2f}s - {ze:.2f}s] by {overlap_sec:.3f}s"
                break

        # Check recording bounds
        if end_sample > filt_data.shape[1]:
            trial_audit_rows.append({
                "participant_id": participant_id,
                "run_id": run_id,
                "trial_idx": tr_idx + 1,
                "cue": cue_code,
                "class_label": class_label,
                "label_name": label_name,
                "onset_sec": round(onset_sec, 3),
                "duration_sec": duration_sec,
                "status": "EXCLUDED",
                "exclusion_reason": "Epoch exceeds recording length",
                "is_finite": False,
                "is_flat": False,
                "c3_ptp_uv": None,
                "cz_ptp_uv": None,
                "c4_ptp_uv": None,
                "max_ptp_uv": None,
            })
            continue

        if has_zero_overlap:
            trial_audit_rows.append({
                "participant_id": participant_id,
                "run_id": run_id,
                "trial_idx": tr_idx + 1,
                "cue": cue_code,
                "class_label": class_label,
                "label_name": label_name,
                "onset_sec": round(onset_sec, 3),
                "duration_sec": duration_sec,
                "status": "EXCLUDED",
                "exclusion_reason": overlap_details,
                "is_finite": False,
                "is_flat": False,
                "c3_ptp_uv": None,
                "cz_ptp_uv": None,
                "c4_ptp_uv": None,
                "max_ptp_uv": None,
            })
            continue

        # Extract 64-channel epoch in microvolts (µV)
        epoch_uv = filt_data[:, onset_sample:end_sample] * 1e6  # (64, 640)

        is_finite = bool(np.all(np.isfinite(epoch_uv)))
        is_flat = bool(np.any(np.std(epoch_uv, axis=1) < 1e-4))

        c3_ptp = float(np.ptp(epoch_uv[c3_idx])) if c3_idx is not None else None
        cz_ptp = float(np.ptp(epoch_uv[cz_idx])) if cz_idx is not None else None
        c4_ptp = float(np.ptp(epoch_uv[c4_idx])) if c4_idx is not None else None
        all_ptps = np.ptp(epoch_uv, axis=1)
        max_ptp = float(np.max(all_ptps))

        if not is_finite:
            status = "EXCLUDED"
            reason = "Non-finite values (NaN or Inf)"
        elif is_flat:
            status = "EXCLUDED"
            reason = "Flatline channel detected (variance < 1e-4)"
        else:
            status = "RETAINED"
            reason = "None"
            retained_epochs.append(epoch_uv)
            retained_labels.append(class_label)

        trial_audit_rows.append({
            "participant_id": participant_id,
            "run_id": run_id,
            "trial_idx": tr_idx + 1,
            "cue": cue_code,
            "class_label": class_label,
            "label_name": label_name,
            "onset_sec": round(onset_sec, 3),
            "duration_sec": duration_sec,
            "status": status,
            "exclusion_reason": reason,
            "is_finite": is_finite,
            "is_flat": is_flat,
            "c3_ptp_uv": round(c3_ptp, 2) if c3_ptp is not None else None,
            "cz_ptp_uv": round(cz_ptp, 2) if cz_ptp is not None else None,
            "c4_ptp_uv": round(c4_ptp, 2) if c4_ptp is not None else None,
            "max_ptp_uv": round(max_ptp, 2),
        })

    return retained_epochs, retained_labels, trial_audit_rows


def run_pipeline() -> dict:
    os.makedirs(EPOCHS_DIR, exist_ok=True)

    print("=" * 84)
    print("MotorMind — 64-Channel Preprocessing Pipeline")
    print(f"Cohort Scope: {len(PARTICIPANTS)} participants ({PARTICIPANTS[0]}..{PARTICIPANTS[-1]})")
    print(f"Runs: {RUNS}")
    print(f"Filter: {PASSBAND_LOW_HZ}–{PASSBAND_HIGH_HZ} Hz zero-phase FIR")
    print(f"Target Output: {TARGET_NPZ_PATH}")
    print("=" * 84)

    all_manifest_rows: list[dict] = []
    all_trial_audit_rows: list[dict] = []
    all_epochs_list: list[np.ndarray] = []
    all_labels_list: list[int] = []
    all_participant_ids: list[str] = []
    all_run_ids: list[str] = []
    canonical_channels: list[str] | None = None

    valid_recordings_count = 0
    t_start_total = time.time()

    for p_idx, participant in enumerate(PARTICIPANTS, 1):
        p_dir = os.path.join(DATASET_DIR, participant)
        if not os.path.isdir(p_dir):
            print(f"[{p_idx}/{len(PARTICIPANTS)}] {participant}: Directory missing, skipping.")
            continue

        for run_id in RUNS:
            edf_file = f"{participant}{run_id}.edf"
            edf_path = os.path.join(p_dir, edf_file)
            rel_path = os.path.join("dataset", participant, edf_file)

            if not os.path.isfile(edf_path):
                all_manifest_rows.append({
                    "participant_id": participant,
                    "run_id": run_id,
                    "filename": edf_file,
                    "file_path": rel_path,
                    "file_size_bytes": 0,
                    "validation_status": "MISSING",
                    "preprocessing_status": "FILE_NOT_FOUND",
                    "sampling_frequency_hz": np.nan,
                    "duration_seconds": np.nan,
                    "n_channels": 0,
                    "retained_trials": 0,
                    "excluded_trials": 0,
                    "exclusion_reasons": "File missing on disk",
                })
                continue

            file_size_bytes = os.path.getsize(edf_path)

            # Step 1: Strict EDF validation
            try:
                val_info = validate_edf_recording(edf_path)
            except Exception as exc:
                val_info = {"valid": False, "status": f"EXCEPTION: {exc}", "error": str(exc)}

            if not val_info.get("valid", False):
                reason = val_info.get("error", val_info.get("status", "Validation failed"))
                all_manifest_rows.append({
                    "participant_id": participant,
                    "run_id": run_id,
                    "filename": edf_file,
                    "file_path": rel_path,
                    "file_size_bytes": file_size_bytes,
                    "validation_status": val_info.get("status", "INVALID"),
                    "preprocessing_status": "FAILED_VALIDATION",
                    "sampling_frequency_hz": val_info.get("sfreq", np.nan),
                    "duration_seconds": val_info.get("duration_sec", np.nan),
                    "n_channels": val_info.get("n_channels", np.nan),
                    "retained_trials": 0,
                    "excluded_trials": 0,
                    "exclusion_reasons": str(reason),
                })
                continue

            valid_recordings_count += 1

            # Step 2: Load and sanitize all channels
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw = sanitize_channel_names(raw)
            sfreq = float(raw.info["sfreq"])
            duration_sec = float(raw.n_times / sfreq)
            ch_names = list(raw.ch_names)

            if canonical_channels is None:
                canonical_channels = ch_names
                print(f"  Initialized canonical 64-channel montage: {len(canonical_channels)} channels.")
            elif ch_names != canonical_channels:
                raise ValueError(f"Channel order mismatch in {edf_file} vs canonical montage.")

            # Step 3: Detect zero flatline intervals on unfiltered data
            raw_data = raw.get_data()
            zero_blocks = detect_zero_intervals(raw_data, sfreq=sfreq, min_duration_sec=0.1)
            del raw_data

            # Step 4: Continuous 8-30 Hz zero-phase FIR filtering across all 64 channels
            raw_filt = filter_eeg_data(raw, l_freq=PASSBAND_LOW_HZ, h_freq=PASSBAND_HIGH_HZ)

            # Step 5: Extract 64-channel epochs
            rec_epochs, rec_labels, rec_trial_rows = extract_64ch_epochs(
                raw_filt=raw_filt,
                zero_intervals=zero_blocks,
                participant_id=participant,
                run_id=run_id,
                channel_names=canonical_channels,
            )

            n_retained = len(rec_epochs)
            n_excluded = sum(1 for row in rec_trial_rows if row.get("status") == "EXCLUDED")
            exclusion_reasons = sorted({
                row.get("exclusion_reason", "")
                for row in rec_trial_rows
                if row.get("status") == "EXCLUDED" and row.get("exclusion_reason")
            })

            for epoch, label in zip(rec_epochs, rec_labels):
                all_epochs_list.append(epoch)
                all_labels_list.append(int(label))
                all_participant_ids.append(participant)
                all_run_ids.append(run_id)

            all_trial_audit_rows.extend(rec_trial_rows)
            all_manifest_rows.append({
                "participant_id": participant,
                "run_id": run_id,
                "filename": edf_file,
                "file_path": rel_path,
                "file_size_bytes": file_size_bytes,
                "validation_status": "VALID",
                "preprocessing_status": "SUCCESS",
                "sampling_frequency_hz": sfreq,
                "duration_seconds": round(duration_sec, 2),
                "n_channels": len(canonical_channels),
                "retained_trials": n_retained,
                "excluded_trials": n_excluded,
                "exclusion_reasons": "; ".join(exclusion_reasons) if exclusion_reasons else "None",
            })

        print(f"  Processed {participant} ({p_idx}/{len(PARTICIPANTS)}) - cumulative trials: {len(all_epochs_list)}")

    if not all_epochs_list:
        raise RuntimeError("No valid epochs extracted.")

    # Consolidate arrays
    X_all = np.asarray(all_epochs_list, dtype=np.float64)
    y_all = np.asarray(all_labels_list, dtype=np.int64)
    y_labels_all = np.asarray([LABEL_NAMES[int(v)] for v in y_all], dtype="<U20")
    p_ids_all = np.asarray(all_participant_ids, dtype="<U4")
    r_ids_all = np.asarray(all_run_ids, dtype="<U3")

    # Save to dedicated 64-channel paths
    np.savez_compressed(
        TARGET_NPZ_PATH,
        X=X_all,
        y=y_all,
        y_labels=y_labels_all,
        participant_ids=p_ids_all,
        run_ids=r_ids_all,
        channel_names=np.asarray(canonical_channels, dtype="<U8"),
        sfreq=float(SAMPLING_FREQ_HZ),
        epoch_duration_sec=float(EPOCH_DURATION_SEC),
        units="uV",
    )
    print(f"\n[SAVED 64-CH DATASET] {TARGET_NPZ_PATH}")
    print(f"  X shape: {X_all.shape}")
    print(f"  y shape: {y_all.shape}")
    print(f"  Channels ({len(canonical_channels)}): {canonical_channels[:5]} ... {canonical_channels[-5:]}")

    # Save metadata
    df_trials = pd.DataFrame(all_trial_audit_rows)
    df_trials.to_csv(TARGET_TRIAL_META_PATH, index=False)
    print(f"[SAVED TRIAL METADATA] {TARGET_TRIAL_META_PATH} ({len(df_trials)} rows)")

    df_manifest = pd.DataFrame(all_manifest_rows)
    df_manifest.to_csv(TARGET_MANIFEST_PATH, index=False)
    print(f"[SAVED MANIFEST] {TARGET_MANIFEST_PATH} ({len(df_manifest)} rows)")

    total_time = time.time() - t_start_total
    print(f"\nPipeline finished in {total_time:.1f}s. Valid recordings: {valid_recordings_count}/210.")

    return {
        "n_recordings_valid": valid_recordings_count,
        "n_trials": len(y_all),
        "shape": list(X_all.shape),
        "channels": canonical_channels,
        "n_participants": len(np.unique(p_ids_all)),
    }


if __name__ == "__main__":
    run_pipeline()
