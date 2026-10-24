"""
MotorMind — Consolidated Dataset Verification Suite
Checks the generated NPZ, trial metadata, and recording manifest without
assuming a fixed trial count or fixed cohort size.
"""
import os
import hashlib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "output", "epochs", "motormind_all_participants_epochs.npz")
TRIAL_META_PATH = os.path.join(BASE_DIR, "output", "epochs", "motormind_trial_metadata.csv")
MANIFEST_PATH = os.path.join(BASE_DIR, "output", "dataset_manifest.csv")

EXPECTED_CHANNELS = ["C3", "Cz", "C4"]
EXPECTED_SFREQ = 160.0
EXPECTED_DURATION = 4.0
EXPECTED_UNITS = "uV"
EXPECTED_RUNS = ["R04", "R08", "R12"]
EXPECTED_PARTICIPANTS = [f"S{i:03d}" for i in range(1, 71)]
FLATLINE_PTP_UV = 1e-8


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def norm(series):
    return series.astype(str).str.strip().str.upper()


def require_columns(df, columns, source):
    missing = [c for c in columns if c not in df.columns]
    require(not missing, f"{source} is missing required columns: {missing}")


def sort_pid(value):
    value = str(value)
    digits = "".join(c for c in value if c.isdigit())
    return (int(digits) if digits else 10**9, value)


def count_duplicate_epochs(X):
    hashes = [
        hashlib.sha256(np.ascontiguousarray(epoch).view(np.uint8)).hexdigest()
        for epoch in X
    ]
    return len(hashes) - len(set(hashes))


def validate():
    print("=" * 78)
    print("MotorMind: Consolidated Multi-Participant Dataset Verification")
    print("=" * 78)

    for path in (NPZ_PATH, TRIAL_META_PATH, MANIFEST_PATH):
        require(os.path.isfile(path), f"Required file is missing: {path}")

    # 1. Load the dataset safely; no pickle is permitted.
    with np.load(NPZ_PATH, allow_pickle=False) as d:
        required_keys = [
            "X", "y", "y_labels", "participant_ids", "run_ids",
            "channel_names", "sfreq", "epoch_duration_sec", "units",
        ]
        require(not [k for k in required_keys if k not in d.files],
                f"NPZ missing keys: {[k for k in required_keys if k not in d.files]}")
        X = d["X"].copy()
        y = d["y"].copy()
        y_labels = d["y_labels"].astype(str)
        p_ids = d["participant_ids"].astype(str)
        r_ids = d["run_ids"].astype(str)
        channels = [str(x) for x in d["channel_names"].tolist()]
        sfreq = float(d["sfreq"])
        duration = float(d["epoch_duration_sec"])
        units = str(d["units"])

    print(f"NPZ path          : {NPZ_PATH}")
    print(f"File size         : {os.path.getsize(NPZ_PATH) / 1048576:.2f} MB")
    print(f"X shape/dtype     : {X.shape} / {X.dtype}")
    print(f"y shape/dtype     : {y.shape} / {y.dtype}")
    print(f"Channels          : {channels}")
    print(f"Sampling rate     : {sfreq:g} Hz")
    print(f"Epoch duration    : {duration:g} s")
    print(f"Units             : {units}")

    require(X.ndim == 3 and X.shape[0] > 0, f"Invalid/empty X shape: {X.shape}")
    require(X.shape[1] == 3, f"Expected 3 channels; got {X.shape[1]}")
    require(channels == EXPECTED_CHANNELS, f"Unexpected channel order: {channels}")
    require(np.isclose(sfreq, EXPECTED_SFREQ), f"Unexpected sampling rate: {sfreq}")
    require(np.isclose(duration, EXPECTED_DURATION), f"Unexpected epoch duration: {duration}")
    require(X.shape[2] == int(round(sfreq * duration)),
            f"Unexpected samples per epoch: {X.shape[2]}")
    require(units == EXPECTED_UNITS, f"Unexpected units: {units!r}")
    require(y.ndim == 1, f"Expected 1D labels, got {y.shape}")
    require(len(X) == len(y) == len(y_labels) == len(p_ids) == len(r_ids),
            "NPZ array lengths do not match.")
    require(set(np.unique(y).tolist()) == {0, 1},
            f"Expected labels 0 and 1, got {np.unique(y).tolist()}")
    require(np.issubdtype(X.dtype, np.number), f"X must be numeric, got {X.dtype}")
    require(np.isfinite(X).all(), "CRITICAL: NaN or Inf values found in X.")

    label_names = {0: "Left_Fist", 1: "Right_Fist"}
    for label, expected_name in label_names.items():
        observed = set(y_labels[y == label].tolist())
        require(observed == {expected_name},
                f"Label names for class {label} are inconsistent: {observed}")

    print("[PASS] NPZ loaded with allow_pickle=False; shape and metadata checks passed.")
    print(f"[PASS] All {X.size:,} signal values are finite.")

    # Flag essentially flat channel epochs. This is not a 200 µV rejection rule.
    channel_ptp = np.ptp(X, axis=2)
    flat_epochs = np.any(channel_ptp <= FLATLINE_PTP_UV, axis=1)
    require(not flat_epochs.any(),
            f"{int(flat_epochs.sum())} epochs have an effectively flat channel.")
    print(f"[PASS] No channel epochs have PtP <= {FLATLINE_PTP_UV:g} µV.")

    # 2. Trial metadata.
    trials = pd.read_csv(TRIAL_META_PATH)
    require_columns(trials, ["status", "class_label", "participant_id", "run_id"],
                    "Trial metadata CSV")
    statuses = norm(trials["status"])
    retained = trials.loc[statuses == "RETAINED"].reset_index(drop=True)
    excluded = trials.loc[statuses == "EXCLUDED"].reset_index(drop=True)
    unknown_status = sorted(set(statuses) - {"RETAINED", "EXCLUDED"})

    print("\n--- Trial Metadata ---")
    print(f"All audit rows : {len(trials)}")
    print(f"Retained       : {len(retained)}")
    print(f"Excluded       : {len(excluded)}")
    require(not unknown_status, f"Unexpected trial status values: {unknown_status}")
    require(len(retained) == len(X),
            f"Retained metadata rows ({len(retained)}) != NPZ trials ({len(X)}).")

    csv_y = pd.to_numeric(retained["class_label"], errors="coerce").to_numpy()
    require(not np.isnan(csv_y).any(), "Non-numeric class labels in retained metadata.")
    require(np.array_equal(csv_y.astype(np.int64), y),
            "CSV labels and NPZ labels differ in row order.")
    require(np.array_equal(retained["participant_id"].astype(str).to_numpy(), p_ids),
            "CSV participant IDs and NPZ IDs differ in row order.")
    require(np.array_equal(retained["run_id"].astype(str).to_numpy(), r_ids),
            "CSV run IDs and NPZ IDs differ in row order.")
    print("[PASS] Retained trial count, labels, participant IDs, and run IDs match NPZ.")

    # 3. Manifest.
    manifest = pd.read_csv(MANIFEST_PATH)
    required_manifest = [
        "participant_id", "run", "validation_status", "preprocessing_status",
        "retained_trials", "excluded_trials",
    ]
    require_columns(manifest, required_manifest, "Recording manifest CSV")
    validation = norm(manifest["validation_status"])
    processing = norm(manifest["preprocessing_status"])
    n_valid = int((validation == "VALID").sum())
    n_success = int((processing == "SUCCESS").sum())
    n_failed = len(manifest) - n_success
    retained_sum = int(pd.to_numeric(manifest["retained_trials"], errors="coerce").fillna(0).sum())
    excluded_sum = int(pd.to_numeric(manifest["excluded_trials"], errors="coerce").fillna(0).sum())

    require(retained_sum == len(X),
            f"Manifest retained sum ({retained_sum}) != NPZ trials ({len(X)}).")
    require(excluded_sum == len(excluded),
            f"Manifest excluded sum ({excluded_sum}) != excluded metadata rows ({len(excluded)}).")
    dup_manifest = int(manifest.duplicated(["participant_id", "run"], keep=False).sum())
    require(dup_manifest == 0, f"Duplicate participant/run rows in manifest: {dup_manifest}")

    expected_pairs = {
        (p, r) for p in EXPECTED_PARTICIPANTS for r in EXPECTED_RUNS
    }
    actual_pairs = set(zip(manifest["participant_id"].astype(str), manifest["run"].astype(str)))
    missing_pairs = sorted(expected_pairs - actual_pairs)
    extra_pairs = sorted(actual_pairs - expected_pairs)

    print("\n--- Recording Manifest ---")
    print(f"Manifest rows             : {len(manifest)}")
    print(f"Valid EDF rows             : {n_valid}")
    print(f"Successfully preprocessed  : {n_success}")
    print(f"Failed/non-success rows    : {n_failed}")
    print(f"Manifest retained sum      : {retained_sum}")
    print(f"Manifest excluded sum      : {excluded_sum}")
    print("[PASS] Manifest trial sums match NPZ and trial metadata.")

    if missing_pairs:
        print(f"[WARNING] {len(missing_pairs)} expected participant/run pairs absent from manifest.")
        print(f"          Examples: {missing_pairs[:10]}")
    if extra_pairs:
        print(f"[WARNING] {len(extra_pairs)} unexpected participant/run pairs in manifest.")
        print(f"          Examples: {extra_pairs[:10]}")
    if n_failed:
        print(f"[WARNING] {n_failed} manifest rows did not complete successfully.")
    if not missing_pairs and not extra_pairs:
        print(f"[PASS] Manifest covers the configured {len(EXPECTED_PARTICIPANTS)} x {len(EXPECTED_RUNS)} grid.")

    # 4. Trial uniqueness.
    trial_col = next(
        (c for c in ["trial_idx", "trial_index", "trial_number", "trial_num"]
         if c in retained.columns),
        None,
    )
    if trial_col:
        key_cols = ["participant_id", "run_id", trial_col]
        duplicates = retained.duplicated(subset=key_cols, keep=False)
        require(not duplicates.any(),
                f"Duplicate trial keys found using {key_cols}: {int(duplicates.sum())} rows.")
        print(f"[PASS] No duplicate trial keys using {key_cols}.")
    else:
        print("[WARNING] No recognized trial-index column; duplicate trial-key check skipped.")

    duplicate_epochs = count_duplicate_epochs(X)
    if duplicate_epochs:
        print(f"[WARNING] {duplicate_epochs} byte-identical epochs found; review before deciding if erroneous.")
    else:
        print("[PASS] No byte-identical epoch arrays found.")

    # 5. Cohort coverage and classes.
    participants = sorted(set(p_ids.tolist()), key=sort_pid)
    runs = sorted(set(r_ids.tolist()))
    missing_participants = sorted(set(EXPECTED_PARTICIPANTS) - set(participants), key=sort_pid)
    extra_participants = sorted(set(participants) - set(EXPECTED_PARTICIPANTS), key=sort_pid)
    missing_runs = sorted(set(EXPECTED_RUNS) - set(runs))
    extra_runs = sorted(set(runs) - set(EXPECTED_RUNS))

    print("\n--- Cohort Coverage ---")
    print(f"Participants represented: {len(participants)}")
    if participants:
        print(f"Participant range       : {participants[0]} through {participants[-1]}")
        print(f"Participant IDs         : {', '.join(participants)}")
    print(f"Runs represented        : {runs}")
    if missing_participants:
        print(f"[WARNING] Participants with no retained epochs ({len(missing_participants)}): {missing_participants}")
    if extra_participants:
        print(f"[WARNING] Unexpected participant IDs: {extra_participants}")
    if missing_runs:
        print(f"[WARNING] Runs with no retained epochs: {missing_runs}")
    if extra_runs:
        print(f"[WARNING] Unexpected run IDs: {extra_runs}")

    n_left = int((y == 0).sum())
    n_right = int((y == 1).sum())
    print("\n--- Class Balance ---")
    print(f"Left Fist (0) : {n_left} ({100 * n_left / len(y):.2f}%)")
    print(f"Right Fist (1): {n_right} ({100 * n_right / len(y):.2f}%)")

    # PtP across all channels and time points for each epoch.
    epoch_ptp = np.ptp(X, axis=(1, 2))
    n_over_200 = int((epoch_ptp > 200.0).sum())
    print("\n--- Peak-to-Peak Screening ---")
    print(f"Epoch-wide PtP range: {epoch_ptp.min():.2f}–{epoch_ptp.max():.2f} µV")
    print(f"Epoch-wide PtP mean : {epoch_ptp.mean():.2f} µV")
    print(f"Epochs > 200 µV     : {n_over_200}/{len(epoch_ptp)} (screening reference, not automatic rejection)")

    print("\n" + "=" * 78)
    if n_failed or missing_pairs or extra_pairs or missing_participants or extra_participants:
        print("INTERNAL CONSISTENCY CHECKS PASSED; REVIEW COHORT-COVERAGE WARNINGS ABOVE.")
    else:
        print("ALL REQUIRED INTEGRITY CHECKS PASSED.")
    print("=" * 78)

    return {
        "trials": len(X),
        "participants": len(participants),
        "successful_recordings": n_success,
        "failed_recordings": n_failed,
        "missing_participants": missing_participants,
        "missing_recordings": missing_pairs,
        "duplicate_epochs": duplicate_epochs,
    }


if __name__ == "__main__":
    validate()
