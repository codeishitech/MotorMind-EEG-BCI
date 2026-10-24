"""
MotorMind — Batch Preprocessing Pipeline (Participants S001–S070)

Processes PhysioNet EEGMMIDB motor-imagery recordings for runs R04, R08, R12.
Uses the project functions in preprocessing_pipeline.py for validation, filtering,
zero-interval detection, epoch extraction, and trial-level quality auditing.

Outputs:
- output/epochs/motormind_all_participants_epochs.npz
- output/epochs/motormind_trial_metadata.csv
- output/trial_metadata.csv
- output/dataset_manifest.csv
- output/quality_checks/fig8_multi_participant_dataset_summary.png
- output/quality_checks/fig9_multi_participant_signal_quality.png
- output/quality_checks/fig10_grand_average_waveforms.png
- output/quality_checks/multi_participant_quality_report.md

Existing outputs with matching names are backed up before processing starts.
"""

import os
import sys
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing_pipeline import (
    SAMPLING_FREQ_HZ,
    EPOCH_DURATION_SEC,
    TARGET_CHANNELS,
    PASSBAND_LOW_HZ,
    PASSBAND_HIGH_HZ,
    LABEL_LEFT_FIST,
    LABEL_RIGHT_FIST,
    LABEL_NAMES,
    sanitize_channel_names,
    validate_edf_recording,
    detect_zero_intervals,
    filter_eeg_data,
    extract_recording_epochs,
)

# ---------------------------------------------------------------------------
# Directories and cohort configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
EPOCHS_DIR = os.path.join(OUTPUT_DIR, "epochs")
QUALITY_DIR = os.path.join(OUTPUT_DIR, "quality_checks")

os.makedirs(EPOCHS_DIR, exist_ok=True)
os.makedirs(QUALITY_DIR, exist_ok=True)

PARTICIPANTS = [f"S{i:03d}" for i in range(1, 71)]  # S001–S070 inclusive
RUNS = ["R04", "R08", "R12"]
P2P_SCREENING_REFERENCE_UV = 200.0


def backup_existing_outputs():
    """Copy existing pipeline outputs to a timestamped backup folder."""
    relative_paths = [
        os.path.join("epochs", "motormind_all_participants_epochs.npz"),
        os.path.join("epochs", "motormind_trial_metadata.csv"),
        "trial_metadata.csv",
        "dataset_manifest.csv",
        os.path.join("quality_checks", "fig8_multi_participant_dataset_summary.png"),
        os.path.join("quality_checks", "fig9_multi_participant_signal_quality.png"),
        os.path.join("quality_checks", "fig10_grand_average_waveforms.png"),
        os.path.join("quality_checks", "multi_participant_quality_report.md"),
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(OUTPUT_DIR, f"pre_expansion_backup_{stamp}")
    copied = []

    for rel_path in relative_paths:
        source = os.path.join(OUTPUT_DIR, rel_path)
        if os.path.isfile(source):
            destination = os.path.join(backup_dir, rel_path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(rel_path)

    if copied:
        print(f"[BACKUP] Existing outputs copied to: {backup_dir}")
        for item in copied:
            print(f"  - {item}")
    else:
        print("[BACKUP] No prior consolidated outputs found; no backup needed.")

    return backup_dir if copied else None


def _safe_float(value):
    """Return a finite float or None for report formatting."""
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _fmt(value, digits=2, suffix=""):
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:.{digits}f}{suffix}"


def _get_retained_and_excluded(df_trials):
    if df_trials.empty or "status" not in df_trials.columns:
        return df_trials.iloc[0:0].copy(), df_trials.iloc[0:0].copy()
    status = df_trials["status"].astype(str).str.upper()
    return (
        df_trials.loc[status == "RETAINED"].copy(),
        df_trials.loc[status == "EXCLUDED"].copy(),
    )


def _participant_ids_from_data(df_trials, p_ids=None):
    """Use participants actually represented in this batch, in numeric order."""
    found = set()
    if p_ids is not None:
        found.update(str(x) for x in np.asarray(p_ids).tolist())
    if not df_trials.empty and "participant_id" in df_trials.columns:
        found.update(df_trials["participant_id"].dropna().astype(str).tolist())

    def sort_key(pid):
        digits = "".join(ch for ch in pid if ch.isdigit())
        return (int(digits) if digits else 10**9, pid)

    return sorted(found, key=sort_key)


def run_batch_preprocessing():
    backup_existing_outputs()

    print("=" * 80)
    print("MotorMind: Batch EEG Preprocessing Pipeline (S001–S070)")
    print("=" * 80)
    print(f"Target Participants : {len(PARTICIPANTS)} ({PARTICIPANTS[0]} to {PARTICIPANTS[-1]})")
    print(f"Target Runs         : {RUNS} (unilateral motor imagery)")
    print(f"Target Recordings   : {len(PARTICIPANTS) * len(RUNS)} EDF files")
    print(f"Target Channels     : {TARGET_CHANNELS}")
    print(f"Band-Pass Filter    : {PASSBAND_LOW_HZ}–{PASSBAND_HIGH_HZ} Hz")
    print(
        f"Epoch Specifications: {EPOCH_DURATION_SEC} s "
        f"({int(EPOCH_DURATION_SEC * SAMPLING_FREQ_HZ)} samples @ {SAMPLING_FREQ_HZ} Hz)"
    )
    print("=" * 80 + "\n")

    all_epochs_list = []
    all_labels_list = []
    all_participant_ids = []
    all_run_ids = []
    all_trial_audit_rows = []
    all_manifest_rows = []

    total_recordings = len(PARTICIPANTS) * len(RUNS)
    processed_count = 0
    valid_recordings_count = 0

    for p_idx, participant in enumerate(PARTICIPANTS, 1):
        print(f"[{p_idx:02d}/{len(PARTICIPANTS)}] Processing participant {participant}...")

        for run_id in RUNS:
            edf_file = f"{participant}{run_id}.edf"
            edf_path = os.path.join(DATASET_DIR, participant, edf_file)
            rel_path = os.path.relpath(edf_path, BASE_DIR).replace("\\", "/")
            file_size_bytes = os.path.getsize(edf_path) if os.path.isfile(edf_path) else 0
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

            # Step 1: Validate the EDF before loading it.
            try:
                val_info = validate_edf_recording(edf_path)
            except Exception as exc:
                val_info = {
                    "valid": False,
                    "status": "VALIDATION_EXCEPTION",
                    "sfreq": np.nan,
                    "duration_sec": np.nan,
                    "n_channels": np.nan,
                    "t0_count": 0,
                    "t1_count": 0,
                    "t2_count": 0,
                    "error": str(exc),
                }

            if not val_info.get("valid", False):
                reason = val_info.get("error", val_info.get("status", "Validation failed"))
                print(f"    [SKIP] {edf_file}: {val_info.get('status', 'INVALID')} — {reason}")
                all_manifest_rows.append({
                    "participant_id": participant,
                    "run": run_id,
                    "filename": edf_file,
                    "file_path": rel_path,
                    "file_size_bytes": file_size_bytes,
                    "file_size_mb": file_size_mb,
                    "validation_status": val_info.get("status", "INVALID"),
                    "preprocessing_status": "FAILED_VALIDATION",
                    "sampling_frequency_hz": val_info.get("sfreq", np.nan),
                    "duration_seconds": val_info.get("duration_sec", np.nan),
                    "n_channels": val_info.get("n_channels", np.nan),
                    "count_t0": val_info.get("t0_count", 0),
                    "count_t1": val_info.get("t1_count", 0),
                    "count_t2": val_info.get("t2_count", 0),
                    "retained_trials": 0,
                    "excluded_trials": 0,
                    "exclusion_reasons": str(reason),
                    "all_event_counts": (
                        f"T0: {val_info.get('t0_count', 0)}, "
                        f"T1: {val_info.get('t1_count', 0)}, "
                        f"T2: {val_info.get('t2_count', 0)}"
                    ),
                })
                continue

            valid_recordings_count += 1

            try:
                # Step 2: Load and sanitize channel names.
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                raw = sanitize_channel_names(raw)
                sfreq = float(raw.info["sfreq"])
                duration_sec = float(raw.n_times / sfreq)

                # Detect flat zero intervals on the original, unfiltered data.
                raw_data = raw.get_data()
                zero_blocks = detect_zero_intervals(
                    raw_data, sfreq=sfreq, min_duration_sec=0.1
                )
                del raw_data

                # Step 3: Filter the continuous recording.
                raw_filt = filter_eeg_data(
                    raw, l_freq=PASSBAND_LOW_HZ, h_freq=PASSBAND_HIGH_HZ
                )

                # Step 4: Keep the motor-strip channels in the required order.
                available = set(raw_filt.ch_names)
                missing_channels = [ch for ch in TARGET_CHANNELS if ch not in available]
                if missing_channels:
                    raise ValueError(
                        f"Missing target channels after sanitization: {missing_channels}; "
                        f"available channels include {raw_filt.ch_names[:10]}"
                    )
                raw_filt.pick(list(TARGET_CHANNELS))

                # Step 5: Extract and audit epochs.
                rec_epochs, rec_labels, rec_trial_rows = extract_recording_epochs(
                    raw_filt=raw_filt,
                    zero_intervals=zero_blocks,
                    participant_id=participant,
                    run_id=run_id,
                )

                n_retained = len(rec_epochs)
                n_excluded = sum(
                    1 for row in rec_trial_rows
                    if str(row.get("status", "")).upper() == "EXCLUDED"
                )
                exclusion_reasons = sorted({
                    str(row.get("exclusion_reason", "Unspecified"))
                    for row in rec_trial_rows
                    if str(row.get("status", "")).upper() == "EXCLUDED"
                    and row.get("exclusion_reason")
                })
                reasons_str = "; ".join(exclusion_reasons) if exclusion_reasons else "None"

                for epoch, label in zip(rec_epochs, rec_labels):
                    all_epochs_list.append(epoch)
                    all_labels_list.append(int(label))
                    all_participant_ids.append(participant)
                    all_run_ids.append(run_id)

                all_trial_audit_rows.extend(rec_trial_rows)

                all_manifest_rows.append({
                    "participant_id": participant,
                    "run": run_id,
                    "filename": edf_file,
                    "file_path": rel_path,
                    "file_size_bytes": file_size_bytes,
                    "file_size_mb": file_size_mb,
                    "validation_status": "VALID",
                    "preprocessing_status": "SUCCESS",
                    "sampling_frequency_hz": sfreq,
                    "duration_seconds": round(duration_sec, 2),
                    "n_channels": len(raw.ch_names),
                    "count_t0": val_info.get("t0_count", 0),
                    "count_t1": val_info.get("t1_count", 0),
                    "count_t2": val_info.get("t2_count", 0),
                    "retained_trials": n_retained,
                    "excluded_trials": n_excluded,
                    "exclusion_reasons": reasons_str,
                    "all_event_counts": (
                        f"T0: {val_info.get('t0_count', 0)}, "
                        f"T1: {val_info.get('t1_count', 0)}, "
                        f"T2: {val_info.get('t2_count', 0)}"
                    ),
                })

                processed_count += 1
                print(
                    f"    - {run_id}: retained {n_retained}; excluded {n_excluded} "
                    f"(duration {duration_sec:.1f}s)"
                )

            except Exception as exc:
                print(f"    [ERROR] {edf_file}: preprocessing failed: {exc}")
                all_manifest_rows.append({
                    "participant_id": participant,
                    "run": run_id,
                    "filename": edf_file,
                    "file_path": rel_path,
                    "file_size_bytes": file_size_bytes,
                    "file_size_mb": file_size_mb,
                    "validation_status": "VALID",
                    "preprocessing_status": "FAILED_PREPROCESSING",
                    "sampling_frequency_hz": val_info.get("sfreq", np.nan),
                    "duration_seconds": val_info.get("duration_sec", np.nan),
                    "n_channels": val_info.get("n_channels", np.nan),
                    "count_t0": val_info.get("t0_count", 0),
                    "count_t1": val_info.get("t1_count", 0),
                    "count_t2": val_info.get("t2_count", 0),
                    "retained_trials": 0,
                    "excluded_trials": 0,
                    "exclusion_reasons": str(exc),
                    "all_event_counts": (
                        f"T0: {val_info.get('t0_count', 0)}, "
                        f"T1: {val_info.get('t1_count', 0)}, "
                        f"T2: {val_info.get('t2_count', 0)}"
                    ),
                })

    # Do not overwrite consolidated files with an empty dataset.
    if not all_epochs_list:
        raise RuntimeError(
            "No valid epochs were extracted. Existing outputs were backed up if present. "
            "Check dataset paths, EDF validation, channel names, and event annotations."
        )

    # -----------------------------------------------------------------------
    # 6. Consolidate arrays
    # -----------------------------------------------------------------------
    X_all = np.asarray(all_epochs_list, dtype=np.float64)
    y_all = np.asarray(all_labels_list, dtype=np.int64)
    y_labels_all = np.asarray([LABEL_NAMES[int(v)] for v in y_all], dtype="<U20")
    p_ids_all = np.asarray(all_participant_ids, dtype="<U4")
    r_ids_all = np.asarray(all_run_ids, dtype="<U3")

    consolidated_npz_path = os.path.join(
        EPOCHS_DIR, "motormind_all_participants_epochs.npz"
    )
    np.savez_compressed(
        consolidated_npz_path,
        X=X_all,
        y=y_all,
        y_labels=y_labels_all,
        participant_ids=p_ids_all,
        run_ids=r_ids_all,
        channel_names=np.asarray(TARGET_CHANNELS, dtype="<U8"),
        sfreq=float(SAMPLING_FREQ_HZ),
        epoch_duration_sec=float(EPOCH_DURATION_SEC),
        units="uV",
    )
    print(f"\n[Saved consolidated dataset] {consolidated_npz_path}")
    print(f"  X shape: {X_all.shape}")
    print(f"  y shape: {y_all.shape}")

    # -----------------------------------------------------------------------
    # 7–8. Save trial metadata and recording manifest
    # -----------------------------------------------------------------------
    df_trials = pd.DataFrame(all_trial_audit_rows)
    trial_meta_path = os.path.join(EPOCHS_DIR, "motormind_trial_metadata.csv")
    df_trials.to_csv(trial_meta_path, index=False, encoding="utf-8")
    df_trials.to_csv(
        os.path.join(OUTPUT_DIR, "trial_metadata.csv"),
        index=False, encoding="utf-8"
    )

    df_manifest = pd.DataFrame(all_manifest_rows)
    manifest_path = os.path.join(OUTPUT_DIR, "dataset_manifest.csv")
    df_manifest.to_csv(manifest_path, index=False, encoding="utf-8")

    print(f"[Saved trial metadata] {trial_meta_path}")
    print(f"[Saved recording manifest] {manifest_path}")

    # -----------------------------------------------------------------------
    # 9–10. Generate dynamic figures and report
    # -----------------------------------------------------------------------
    print("\nGenerating report-ready 300 DPI figures...")
    generate_qc_figures(df_trials, df_manifest, X_all, y_all, p_ids_all)

    report_path = os.path.join(QUALITY_DIR, "multi_participant_quality_report.md")
    generate_qc_report(report_path, df_trials, df_manifest, X_all, y_all)
    print(f"[Saved QC report] {report_path}")

    # -----------------------------------------------------------------------
    # 11. Self-validation of the saved dataset
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VALIDATING SAVED CONSOLIDATED DATASET (allow_pickle=False)")
    print("=" * 80)

    with np.load(consolidated_npz_path, allow_pickle=False) as loaded:
        X_check = loaded["X"]
        y_check = loaded["y"]
        y_lbl_check = loaded["y_labels"]
        p_check = loaded["participant_ids"]
        r_check = loaded["run_ids"]
        ch_check = loaded["channel_names"].tolist()
        sfreq_check = float(loaded["sfreq"])
        epoch_duration_check = float(loaded["epoch_duration_sec"])
        units_check = str(loaded["units"])

    assert X_check.ndim == 3 and X_check.shape[1] == 3, f"Invalid X shape: {X_check.shape}"
    expected_samples = int(round(epoch_duration_check * sfreq_check))
    assert X_check.shape[2] == expected_samples, "Unexpected samples per epoch"
    assert len(X_check) == len(y_check) == len(p_check) == len(r_check) == len(y_lbl_check)
    assert set(np.unique(y_check).tolist()) == {0, 1}, "Expected both binary labels 0 and 1"
    assert np.isfinite(X_check).all(), "NaN or Inf values found"
    assert ch_check == list(TARGET_CHANNELS), f"Unexpected channel order: {ch_check}"
    assert np.isclose(sfreq_check, SAMPLING_FREQ_HZ), f"Unexpected sampling rate: {sfreq_check}"
    assert units_check == "uV", f"Unexpected amplitude units: {units_check}"

    represented_participants = set(p_check.tolist())
    missing_participants = sorted(set(PARTICIPANTS) - represented_participants)
    if missing_participants:
        print(
            f"[WARNING] {len(missing_participants)} configured participants have no "
            f"retained epochs: {', '.join(missing_participants)}"
        )
    else:
        print(f"[PASS] All {len(PARTICIPANTS)} configured participants are represented.")

    print("[PASS] Dataset loaded with allow_pickle=False.")
    print(f"[PASS] X shape: {X_check.shape}; channels: {ch_check}.")
    print(f"[PASS] Samples per epoch: {X_check.shape[2]} at {sfreq_check:g} Hz.")
    print(
        f"[PASS] Binary classes: left={int(np.sum(y_check == 0))}, "
        f"right={int(np.sum(y_check == 1))}."
    )
    print(f"[PASS] All {X_check.size:,} signal values are finite.")
    print(f"[PASS] Retained epochs: {len(X_check):,}.")
    print(
        f"[SUMMARY] Valid EDFs: {valid_recordings_count}; successfully processed: "
        f"{processed_count}; configured EDFs: {total_recordings}."
    )
    print("=" * 80)

    return df_trials, df_manifest, X_all, y_all


def generate_qc_figures(df_trials, df_manifest, X, y, p_ids):
    """Generate dataset summary, amplitude QC, and grand-average waveform figures."""
    retained_df, excluded_df = _get_retained_and_excluded(df_trials)
    participant_ids = _participant_ids_from_data(df_trials, p_ids)
    n_trials = len(y)
    total_left = int(np.sum(y == 0))
    total_right = int(np.sum(y == 1))
    total_excluded = len(excluded_df)

    # -----------------------------------------------------------------------
    # Figure 8: participant yield, class balance, and run yield
    # -----------------------------------------------------------------------
    fig8 = plt.figure(figsize=(max(16, len(participant_ids) * 0.28), 10), dpi=150)
    gs8 = fig8.add_gridspec(2, 2, height_ratios=[1.3, 1.0])
    ax_part = fig8.add_subplot(gs8[0, :])
    ax_pie = fig8.add_subplot(gs8[1, 0])
    ax_run = fig8.add_subplot(gs8[1, 1])

    left_counts, right_counts = [], []
    for participant in participant_ids:
        sub = retained_df[retained_df["participant_id"].astype(str) == participant]
        labels = pd.to_numeric(sub.get("class_label", pd.Series(dtype=float)), errors="coerce")
        left_counts.append(int((labels == 0).sum()))
        right_counts.append(int((labels == 1).sum()))

    x_idx = np.arange(len(participant_ids))
    ax_part.bar(x_idx, left_counts, width=0.72, label="Left fist (class 0)")
    ax_part.bar(
        x_idx, right_counts, width=0.72, bottom=left_counts,
        label="Right fist (class 1)"
    )
    totals = np.asarray(left_counts) + np.asarray(right_counts)
    for idx, total in enumerate(totals):
        if total:
            ax_part.text(idx, total + max(0.5, totals.max() * 0.01), str(int(total)),
                         ha="center", va="bottom", fontsize=7)
    ax_part.set_title("A. Retained motor-imagery trials per represented participant")
    ax_part.set_xlabel("Participant")
    ax_part.set_ylabel("Retained trials")
    ax_part.set_xticks(x_idx)
    ax_part.set_xticklabels(participant_ids, rotation=90, fontsize=7)
    ymax = max(1, int(totals.max()) if len(totals) else 1)
    ax_part.set_ylim(0, ymax * 1.18 + 1)
    ax_part.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax_part.legend(loc="upper right", fontsize=8)

    if n_trials:
        pie_counts = [total_left, total_right]
        pie_labels = [
            f"Left fist\nn={total_left} ({100 * total_left / n_trials:.1f}%)",
            f"Right fist\nn={total_right} ({100 * total_right / n_trials:.1f}%)",
        ]
        if sum(pie_counts) > 0:
            ax_pie.pie(
                pie_counts, labels=pie_labels, autopct="%1.1f%%",
                startangle=140, wedgeprops=dict(width=0.45, edgecolor="white"),
                textprops=dict(fontsize=9),
            )
        else:
            ax_pie.text(0.5, 0.5, "No retained trials", ha="center", va="center")
    else:
        ax_pie.text(0.5, 0.5, "No retained trials", ha="center", va="center")
    ax_pie.set_title(f"B. Class balance (retained={n_trials}, excluded={total_excluded})")

    if not retained_df.empty and "run_id" in retained_df.columns:
        run_counts = [
            int((retained_df["run_id"].astype(str) == run).sum()) for run in RUNS
        ]
    else:
        run_counts = [0] * len(RUNS)
    ax_run.bar(RUNS, run_counts, width=0.55)
    run_ymax = max(1, max(run_counts, default=0))
    for idx, count in enumerate(run_counts):
        pct = 100 * count / n_trials if n_trials else 0
        ax_run.text(idx, count + run_ymax * 0.025, f"{count}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=8)
    ax_run.set_title("C. Retained trials by run")
    ax_run.set_xlabel("PhysioNet run")
    ax_run.set_ylabel("Retained trials")
    ax_run.set_ylim(0, run_ymax * 1.2 + 1)
    ax_run.grid(True, linestyle="--", alpha=0.35, axis="y")

    fig8.suptitle(
        f"MotorMind: Dataset Summary ({len(participant_ids)} represented participants, "
        f"{n_trials} retained trials)", fontsize=13, fontweight="bold"
    )
    fig8.tight_layout(rect=[0, 0, 1, 0.96])
    fig8_path = os.path.join(QUALITY_DIR, "fig8_multi_participant_dataset_summary.png")
    fig8.savefig(fig8_path, dpi=300, bbox_inches="tight")
    plt.close(fig8)
    print(f"  [Saved Figure 8] {fig8_path}")

    # -----------------------------------------------------------------------
    # Figure 9: peak-to-peak distributions and participant-level amplitude
    # -----------------------------------------------------------------------
    fig9, (ax_box, ax_subject) = plt.subplots(
        1, 2, figsize=(max(16, len(participant_ids) * 0.28), 6), dpi=150
    )
    channel_columns = ["c3_ptp_uv", "cz_ptp_uv", "c4_ptp_uv"]
    box_data = []
    for column in channel_columns:
        if column in retained_df.columns:
            values = pd.to_numeric(retained_df[column], errors="coerce").dropna().to_numpy()
        else:
            values = np.asarray([])
        box_data.append(values)

    if any(len(values) for values in box_data):
        ax_box.boxplot(
            box_data, tick_labels=["C3", "Cz", "C4"], patch_artist=True,
            widths=0.5, showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.35),
        )
    else:
        ax_box.text(0.5, 0.5, "No channel PtP metrics available",
                    transform=ax_box.transAxes, ha="center")
    ax_box.axhline(
        P2P_SCREENING_REFERENCE_UV, linestyle="--", linewidth=1.4,
        label=f"Screening reference ({P2P_SCREENING_REFERENCE_UV:g} µV)"
    )
    channel_values = np.concatenate([v for v in box_data if len(v)]) if any(
        len(v) for v in box_data
    ) else np.asarray([P2P_SCREENING_REFERENCE_UV])
    box_upper = max(P2P_SCREENING_REFERENCE_UV * 1.15, float(np.nanpercentile(channel_values, 98)) * 1.15)
    ax_box.set_ylim(bottom=0, top=box_upper)
    ax_box.set_title(f"A. Peak-to-peak amplitude by channel (n={n_trials} trials)")
    ax_box.set_xlabel("Electrode channel")
    ax_box.set_ylabel("Peak-to-peak amplitude (µV)")
    ax_box.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax_box.legend(fontsize=8)

    means, stds = [], []
    for participant in participant_ids:
        sub = retained_df[retained_df["participant_id"].astype(str) == participant]
        values = pd.to_numeric(sub.get("max_ptp_uv", pd.Series(dtype=float)), errors="coerce").dropna()
        means.append(float(values.mean()) if len(values) else np.nan)
        stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)

    sx = np.arange(len(participant_ids))
    if participant_ids:
        ax_subject.errorbar(
            sx, means, yerr=stds, fmt="o-", capsize=2, linewidth=1,
            markersize=3.5, label="Mean ± SD per participant"
        )
        ax_subject.set_xticks(sx)
        ax_subject.set_xticklabels(participant_ids, rotation=90, fontsize=7)
    ax_subject.axhline(
        P2P_SCREENING_REFERENCE_UV, linestyle="--", linewidth=1.2,
        label=f"Screening reference ({P2P_SCREENING_REFERENCE_UV:g} µV)"
    )
    valid_means = np.asarray(means, dtype=float)
    valid_means = valid_means[np.isfinite(valid_means)]
    valid_stds = np.asarray(stds, dtype=float)
    valid_stds = valid_stds[np.isfinite(valid_stds)]
    subject_max = max(
        [P2P_SCREENING_REFERENCE_UV * 1.15]
        + (list(valid_means + valid_stds) if len(valid_means) else [])
    )
    ax_subject.set_ylim(bottom=0, top=subject_max * 1.15 + 1)
    ax_subject.set_title("B. Participant-level maximum channel PtP")
    ax_subject.set_xlabel("Participant")
    ax_subject.set_ylabel("Maximum channel PtP (µV)")
    ax_subject.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax_subject.legend(fontsize=8)

    fig9.suptitle("MotorMind: Signal-Quality Screening and Participant Variability",
                  fontsize=13, fontweight="bold")
    fig9.tight_layout(rect=[0, 0, 1, 0.95])
    fig9_path = os.path.join(QUALITY_DIR, "fig9_multi_participant_signal_quality.png")
    fig9.savefig(fig9_path, dpi=300, bbox_inches="tight")
    plt.close(fig9)
    print(f"  [Saved Figure 9] {fig9_path}")

    # -----------------------------------------------------------------------
    # Figure 10: grand-average voltage waveforms (not ERD)
    # -----------------------------------------------------------------------
    fig10, axes = plt.subplots(1, len(TARGET_CHANNELS), figsize=(16, 5), dpi=150)
    axes = np.atleast_1d(axes)
    time_vec = np.arange(X.shape[2]) / float(SAMPLING_FREQ_HZ)
    left_mask = y == 0
    right_mask = y == 1

    for ch_idx, channel in enumerate(TARGET_CHANNELS):
        ax = axes[ch_idx]
        for mask, label, linestyle in [
            (left_mask, "Imagined left fist (class 0)", "-"),
            (right_mask, "Imagined right fist (class 1)", "--"),
        ]:
            data = X[mask, ch_idx, :]
            if len(data):
                mean = np.mean(data, axis=0)
                sem = (
                    np.std(data, axis=0, ddof=1) / np.sqrt(len(data))
                    if len(data) > 1 else np.zeros_like(mean)
                )
                ax.plot(time_vec, mean, linestyle=linestyle, linewidth=1.4,
                        label=f"{label}, n={len(data)}")
                ax.fill_between(time_vec, mean - sem, mean + sem, alpha=0.18)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
        ax.set_title(f"{channel}: grand-average voltage waveform")
        ax.set_xlabel("Time from event (seconds)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_xlim(time_vec[0] if len(time_vec) else 0, EPOCH_DURATION_SEC)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7)

    fig10.suptitle(
        f"MotorMind: Grand-Average Filtered Voltage Waveforms "
        f"({len(X)} retained trials; {len(participant_ids)} represented participants)",
        fontsize=12, fontweight="bold"
    )
    fig10.tight_layout(rect=[0, 0, 1, 0.93])
    fig10_path = os.path.join(QUALITY_DIR, "fig10_grand_average_waveforms.png")
    fig10.savefig(fig10_path, dpi=300, bbox_inches="tight")
    plt.close(fig10)
    print(f"  [Saved Figure 10] {fig10_path}")


def generate_qc_report(report_path, df_trials, df_manifest, X, y):
    """Write a data-driven Markdown quality-control report."""
    retained_df, excluded_df = _get_retained_and_excluded(df_trials)
    participant_ids = _participant_ids_from_data(df_trials)
    n_trials = len(y)
    n_left = int(np.sum(y == 0))
    n_right = int(np.sum(y == 1))
    n_expected = len(PARTICIPANTS) * len(RUNS)

    if not df_manifest.empty and "preprocessing_status" in df_manifest.columns:
        statuses = df_manifest["preprocessing_status"].astype(str)
        n_success = int((statuses == "SUCCESS").sum())
        n_failed_validation = int((statuses == "FAILED_VALIDATION").sum())
        n_failed_preprocessing = int((statuses == "FAILED_PREPROCESSING").sum())
        n_valid = int((df_manifest["validation_status"].astype(str) == "VALID").sum())
    else:
        n_success = n_failed_validation = n_failed_preprocessing = n_valid = 0

    if not df_manifest.empty and "participant_id" in df_manifest.columns:
        participants_with_successful_recording = set(
            df_manifest.loc[
                df_manifest.get("preprocessing_status", pd.Series(index=df_manifest.index, dtype=str))
                .astype(str).eq("SUCCESS"),
                "participant_id"
            ].dropna().astype(str)
        )
    else:
        participants_with_successful_recording = set()

    n_excluded = len(excluded_df)
    total_cues = len(df_trials)
    retention_pct = 100 * n_trials / total_cues if total_cues else 0.0
    left_pct = 100 * n_left / n_trials if n_trials else 0.0
    right_pct = 100 * n_right / n_trials if n_trials else 0.0

    # Determine available amplitude columns without assuming all metadata fields exist.
    amplitude_stats = {}
    for col in ["c3_ptp_uv", "cz_ptp_uv", "c4_ptp_uv", "max_ptp_uv"]:
        if col in retained_df.columns:
            vals = pd.to_numeric(retained_df[col], errors="coerce").dropna()
            if len(vals):
                amplitude_stats[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "over_threshold": int((vals > P2P_SCREENING_REFERENCE_UV).sum()),
                    "count": int(len(vals)),
                }

    p_rows = []
    for participant in PARTICIPANTS:
        retained_sub = retained_df[
            retained_df.get("participant_id", pd.Series(dtype=str)).astype(str) == participant
        ] if not retained_df.empty else retained_df
        excluded_sub = excluded_df[
            excluded_df.get("participant_id", pd.Series(dtype=str)).astype(str) == participant
        ] if not excluded_df.empty else excluded_df
        manifest_sub = df_manifest[
            df_manifest.get("participant_id", pd.Series(dtype=str)).astype(str) == participant
        ] if not df_manifest.empty else df_manifest

        left_n = int((pd.to_numeric(
            retained_sub.get("class_label", pd.Series(dtype=float)), errors="coerce"
        ) == 0).sum())
        right_n = int((pd.to_numeric(
            retained_sub.get("class_label", pd.Series(dtype=float)), errors="coerce"
        ) == 1).sum())
        n_retained_p = len(retained_sub)
        n_excluded_p = len(excluded_sub)
        n_cues_p = n_retained_p + n_excluded_p
        runs_successful = int(
            (manifest_sub.get("preprocessing_status", pd.Series(dtype=str)).astype(str) == "SUCCESS").sum()
        )
        mean_ptp = (
            pd.to_numeric(retained_sub["max_ptp_uv"], errors="coerce").mean()
            if "max_ptp_uv" in retained_sub.columns and n_retained_p else np.nan
        )
        participant_status = (
            "PASS" if n_retained_p else
            ("NO_RETAINED_TRIALS" if runs_successful else "NO_SUCCESSFUL_RECORDING")
        )
        p_rows.append(
            f"| `{participant}` | {runs_successful} | {n_cues_p} | {n_retained_p} | "
            f"{left_n} | {right_n} | {n_excluded_p} | {_fmt(mean_ptp, 2, ' µV')} | "
            f"{participant_status} |"
        )

    if amplitude_stats:
        max_ptp = amplitude_stats.get("max_ptp_uv")
        if max_ptp:
            max_ptp_summary = (
                f"- **Mean maximum-channel PtP:** {_fmt(max_ptp['mean'], 2, ' µV')} "
                f"(SD {_fmt(max_ptp['std'], 2, ' µV')})\n"
                f"- **Minimum/maximum maximum-channel PtP:** "
                f"{_fmt(max_ptp['min'], 2, ' µV')} / {_fmt(max_ptp['max'], 2, ' µV')}\n"
                f"- **Trials with max-channel PtP above {P2P_SCREENING_REFERENCE_UV:g} µV:** "
                f"{max_ptp['over_threshold']} of {max_ptp['count']} "
                f"({100 * max_ptp['over_threshold'] / max_ptp['count']:.2f}%)"
            )
        else:
            max_ptp_summary = "Maximum-channel PtP summary is unavailable in trial metadata."
        channel_lines = []
        for col, channel in [
            ("c3_ptp_uv", "C3"), ("cz_ptp_uv", "Cz"), ("c4_ptp_uv", "C4")
        ]:
            if col in amplitude_stats:
                stat = amplitude_stats[col]
                channel_lines.append(
                    f"- **{channel} mean PtP:** {_fmt(stat['mean'], 2, ' µV')} "
                    f"(SD {_fmt(stat['std'], 2, ' µV')}; n={stat['count']})"
                )
        channel_summary = "\n".join(channel_lines) if channel_lines else "Per-channel PtP summaries unavailable."
    else:
        max_ptp_summary = "Peak-to-peak statistics were not available in the retained trial metadata."
        channel_summary = "Per-channel PtP summaries unavailable."

    report_text = f"""# MotorMind: Multi-Participant EEG Preprocessing and Quality-Control Report

**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Configured cohort:** `{PARTICIPANTS[0]}` through `{PARTICIPANTS[-1]}` ({len(PARTICIPANTS)} participants)  
**Task runs:** {", ".join(f"`{run}`" for run in RUNS)} (unilateral motor-imagery runs)  
**Target recordings:** {n_expected} EDF files  
**Target channels:** {", ".join(f"`{ch}`" for ch in TARGET_CHANNELS)}  
**Sampling rate:** {SAMPLING_FREQ_HZ:g} Hz  
**Band-pass filter:** {PASSBAND_LOW_HZ:g}–{PASSBAND_HIGH_HZ:g} Hz, using the project filtering function  
**Epoch duration:** {EPOCH_DURATION_SEC:g} seconds ({int(round(EPOCH_DURATION_SEC * SAMPLING_FREQ_HZ))} samples)  
**Amplitude units:** microvolts (µV)

---

## 1. Executive summary

| Metric | Result |
|---|---:|
| Configured EDF recordings | {n_expected} |
| EDFs validated as valid | {n_valid} |
| Successfully preprocessed recordings | {n_success} |
| Validation failures | {n_failed_validation} |
| Preprocessing failures | {n_failed_preprocessing} |
| Participants with a successful recording | {len(participants_with_successful_recording)} |
| Participants represented by retained epochs | {retained_df['participant_id'].astype(str).nunique() if not retained_df.empty and 'participant_id' in retained_df.columns else 0} |
| Trial audit rows / candidate cues | {total_cues} |
| Retained trials | {n_trials} |
| Excluded trials | {n_excluded} |
| Retention rate | {retention_pct:.2f}% |
| Left-fist trials (class 0) | {n_left} ({left_pct:.2f}%) |
| Right-fist trials (class 1) | {n_right} ({right_pct:.2f}%) |

The report statistics and figures are computed from the dataset produced by this run. They are not fixed to the earlier 19-participant baseline.

---

## 2. Participant-level trial yield

| Participant | Successful runs | Audited cues | Retained | Left fist (0) | Right fist (1) | Excluded | Mean max PtP | Status |
|:---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(p_rows)}

---

## 3. Signal quality checks

The pipeline's epoch extraction and QC checks are delegated to `preprocessing_pipeline.py`. The following summary is calculated from retained-trial metadata where those fields are available.

{max_ptp_summary}

{channel_summary}

**Interpretation of the {P2P_SCREENING_REFERENCE_UV:g} µV reference:** this is a screening reference, not an automatic rejection rule unless the preprocessing function explicitly implements it that way. The report records observed values rather than silently removing trials based only on this threshold.

The consolidated arrays are checked for finite numeric values. Epoch dimensions, label values, channel order, sampling rate, and units are also validated before the run is considered complete.

---

## 4. Processing and audit details

- **Raw data validation:** each configured EDF path is checked with `validate_edf_recording`.
- **Zero-interval screening:** candidate zero-valued intervals are detected from the original recording before filtering.
- **Filtering:** continuous data are passed through `filter_eeg_data` using the configured 8–30 Hz passband.
- **Channel selection:** only {", ".join(TARGET_CHANNELS)} are retained, in that order.
- **Epoch extraction:** `extract_recording_epochs` creates task epochs and writes trial-level audit rows.
- **Failure handling:** missing/invalid EDFs and per-recording preprocessing exceptions are recorded in the manifest, and processing continues with other recordings.
- **Participant coverage:** a participant with no retained trials is reported as missing/underrepresented; this does not by itself invalidate otherwise usable participants.
- **Model evaluation:** this preprocessing script does not train or evaluate a classifier.

---

## 5. Output files

| Artifact | Relative path | Purpose |
|---|---|---|
| Consolidated epochs | `output/epochs/motormind_all_participants_epochs.npz` | Signal array `X`, labels `y`, participant/run IDs, channels, sampling rate, epoch duration, units |
| Trial metadata | `output/epochs/motormind_trial_metadata.csv` | Per-trial audit catalog |
| Trial metadata convenience copy | `output/trial_metadata.csv` | Copy of the trial audit catalog |
| Recording manifest | `output/dataset_manifest.csv` | Per-recording validation, processing status, event counts, and retained/excluded counts |
| Figure 8 | `output/quality_checks/fig8_multi_participant_dataset_summary.png` | Participant yield, class balance, run distribution |
| Figure 9 | `output/quality_checks/fig9_multi_participant_signal_quality.png` | PtP amplitude distributions and participant-level summary |
| Figure 10 | `output/quality_checks/fig10_grand_average_waveforms.png` | Grand-average filtered voltage waveforms by channel and class |
| QC report | `output/quality_checks/multi_participant_quality_report.md` | This report |

All three figures are saved at 300 DPI.

---

## 6. Preservation and interpretation notes

- Existing consolidated outputs are copied to a timestamped `output/pre_expansion_backup_YYYYMMDD_HHMMSS/` folder before this pipeline writes new outputs.
- The single-participant epoch file `output/epochs/eeg_motor_imagery_epochs.npz` is not written by this script.
- Figure 10 shows **grand-average voltage waveforms**, not event-related desynchronization (ERD). ERD requires a separate power/time-frequency analysis.
- The consolidated dataset is intended for offline research and coursework; this script does not create a real-time BCI or a medical device.
"""
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)


if __name__ == "__main__":
    run_batch_preprocessing()
