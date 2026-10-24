"""
MotorMind — Reusable EEG Preprocessing Pipeline
===============================================
Provides standardized functions for:
  - Channel name sanitation (10-10 international standard)
  - Strict EDF validation (header records vs. file size, sampling rate, channels)
  - Zero-signal (flatline) interval scanning across continuous channels
  - Zero-phase 8–30 Hz FIR band-pass filtering (sensorimotor Mu/Beta rhythm isolation)
  - Run-specific motor-imagery trial extraction (Left Fist vs. Right Fist)
  - Epoch-level quality control (finite values, flatlines, peak-to-peak amplitude)
"""

import os
import struct
import numpy as np
import mne
from typing import Dict, List, Tuple, Optional, Any

# Standardized parameters
SAMPLING_FREQ_HZ = 160.0
EPOCH_DURATION_SEC = 4.0
TARGET_CHANNELS = ["C3", "Cz", "C4"]
PASSBAND_LOW_HZ = 8.0
PASSBAND_HIGH_HZ = 30.0

# Class labels
LABEL_LEFT_FIST = 0
LABEL_RIGHT_FIST = 1
LABEL_NAMES = {LABEL_LEFT_FIST: "Left_Fist", LABEL_RIGHT_FIST: "Right_Fist"}


def sanitize_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Standardizes PhysioNet channel names (e.g., 'C3..', 'Fc5.', 'Cz.')
    to international 10-10 standards ('C3', 'Cz', 'C4').
    """
    clean_map = {}
    for name in raw.ch_names:
        clean = name.strip().rstrip('.').upper()
        if clean.endswith('Z'):
            clean = clean[:-1] + 'z'
        clean_map[name] = clean
    raw.rename_channels(clean_map)
    return raw


def inspect_edf_header(file_path: str) -> Dict[str, Any]:
    """
    Reads the low-level binary EDF header without loading data into RAM.
    Verifies header integrity, number of records, and expected total file size.
    """
    if not os.path.exists(file_path):
        return {"exists": False, "size_bytes": 0, "header_valid": False, "error": "File not found"}

    actual_size = os.path.getsize(file_path)
    if actual_size < 256:
        return {"exists": True, "size_bytes": actual_size, "header_valid": False, "error": "File too small (<256 bytes)"}

    try:
        with open(file_path, "rb") as f:
            hdr = f.read(256)
            header_bytes = int(hdr[184:192].strip())
            num_records = int(hdr[236:244].strip())
            rec_dur = float(hdr[244:252].strip())
            num_signals = int(hdr[252:256].strip())

            # Read sample counts per record for each signal
            # Offset to signal samples: 256 + ns*16 (labels) + ns*80 (transducer) + ns*8 (phys dim) +
            # ns*8 (phys min) + ns*8 (phys max) + ns*8 (dig min) + ns*8 (dig max) + ns*80 (prefiltering)
            offset = 256 + num_signals * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
            f.seek(offset)
            samples_per_rec = [int(f.read(8).strip()) for _ in range(num_signals)]
            bytes_per_rec = sum(s * 2 for s in samples_per_rec)
            expected_size = header_bytes + num_records * bytes_per_rec

        size_matches = (actual_size == expected_size)
        return {
            "exists": True,
            "size_bytes": actual_size,
            "header_bytes": header_bytes,
            "num_records": num_records,
            "rec_duration_sec": rec_dur,
            "num_signals": num_signals,
            "expected_size_bytes": expected_size,
            "size_matches": size_matches,
            "header_valid": True,
            "error": None if size_matches else f"Truncated file: size {actual_size} != header expected {expected_size}"
        }
    except Exception as e:
        return {
            "exists": True,
            "size_bytes": actual_size,
            "header_valid": False,
            "error": f"Corrupt EDF header: {str(e)}"
        }


def validate_edf_recording(file_path: str) -> Dict[str, Any]:
    """
    Validates EDF recording before full preprocessing:
      1. File existence and binary header consistency
      2. MNE metadata check (preload=False)
      3. Sampling frequency (must equal 160.0 Hz)
      4. Motor channels presence (C3, Cz, C4)
      5. Task annotation availability (T1 and T2)
    """
    header_info = inspect_edf_header(file_path)
    if not header_info.get("header_valid") or not header_info.get("size_matches"):
        return {
            "valid": False,
            "status": "CORRUPT_OR_TRUNCATED",
            "error": header_info.get("error"),
            "sfreq": None,
            "duration_sec": None,
            "n_channels": None,
            "t0_count": 0,
            "t1_count": 0,
            "t2_count": 0
        }

    try:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        sfreq = float(raw.info["sfreq"])
        n_times = raw.n_times
        duration_sec = float(n_times / sfreq)
        raw_channels = [c.strip().rstrip('.').upper().replace('CZ', 'Cz') for c in raw.ch_names]
        n_channels = len(raw_channels)

        has_c3 = "C3" in raw_channels
        has_cz = "Cz" in raw_channels
        has_c4 = "C4" in raw_channels
        has_motor = has_c3 and has_cz and has_c4

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        id_to_desc = {v: k for k, v in event_id.items()}

        counts = {"T0": 0, "T1": 0, "T2": 0}
        for ev in events:
            desc = id_to_desc.get(ev[2], "UNKNOWN")
            if desc in counts:
                counts[desc] += 1

        issues = []
        if sfreq != SAMPLING_FREQ_HZ:
            issues.append(f"Unexpected sfreq ({sfreq} Hz)")
        if not has_motor:
            issues.append(f"Missing motor channels (C3:{has_c3}, Cz:{has_cz}, C4:{has_c4})")
        if counts["T1"] == 0 or counts["T2"] == 0:
            issues.append(f"Missing motor-imagery events (T1:{counts['T1']}, T2:{counts['T2']})")

        if issues:
            return {
                "valid": False,
                "status": "VALIDATION_WARNING: " + "; ".join(issues),
                "error": "; ".join(issues),
                "sfreq": sfreq,
                "duration_sec": round(duration_sec, 2),
                "n_channels": n_channels,
                "t0_count": counts["T0"],
                "t1_count": counts["T1"],
                "t2_count": counts["T2"]
            }

        return {
            "valid": True,
            "status": "VALID",
            "error": None,
            "sfreq": sfreq,
            "duration_sec": round(duration_sec, 2),
            "n_channels": n_channels,
            "t0_count": counts["T0"],
            "t1_count": counts["T1"],
            "t2_count": counts["T2"]
        }
    except Exception as e:
        return {
            "valid": False,
            "status": f"READ_ERROR: {str(e)}",
            "error": str(e),
            "sfreq": None,
            "duration_sec": None,
            "n_channels": None,
            "t0_count": 0,
            "t1_count": 0,
            "t2_count": 0
        }


def detect_zero_intervals(data: np.ndarray, sfreq: float, min_duration_sec: float = 0.1) -> List[Tuple[float, float, float, int]]:
    """
    Detects continuous intervals where all channels are simultaneously 0.0 V.
    Returns list of tuples: (start_sec, end_sec, duration_sec, sample_count).
    """
    all_zero_mask = np.all(data == 0.0, axis=0)
    if not np.any(all_zero_mask):
        return []

    zero_indices = np.where(all_zero_mask)[0]
    diffs = np.diff(zero_indices)
    breaks = np.where(diffs > 1)[0]
    starts = [zero_indices[0]] + [zero_indices[b + 1] for b in breaks]
    ends = [zero_indices[b] for b in breaks] + [zero_indices[-1]]

    blocks = []
    for s, e in zip(starts, ends):
        t_s = s / sfreq
        t_e = (e + 1) / sfreq
        dur = t_e - t_s
        count = e - s + 1
        if dur >= min_duration_sec:
            blocks.append((t_s, t_e, dur, count))
    return blocks


def filter_eeg_data(raw: mne.io.BaseRaw,
                    l_freq: float = PASSBAND_LOW_HZ,
                    h_freq: float = PASSBAND_HIGH_HZ) -> mne.io.BaseRaw:
    """
    Applies zero-phase windowed sinc FIR band-pass filtering (8–30 Hz).
    Preserves phase linearity and avoids phase distortion across sensorimotor rhythms.
    """
    raw_filt = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        phase="zero",
        verbose=False
    )
    return raw_filt


def extract_recording_epochs(raw_filt: mne.io.BaseRaw,
                             zero_intervals: List[Tuple[float, float, float, int]],
                             participant_id: str,
                             run_id: str) -> Tuple[List[np.ndarray], List[int], List[Dict[str, Any]]]:
    """
    Extracts 4.0-second epochs (640 samples @ 160 Hz) for C3, Cz, C4.
    Checks for overlap with zero-signal intervals and evaluates signal quality.
    
    Returns:
      - retained_epochs: list of np.ndarray of shape (3, 640) in µV
      - retained_labels: list of binary int (0 = Left_Fist, 1 = Right_Fist)
      - trial_audit_rows: list of dicts with trial-level audit metadata
    """
    sfreq = float(raw_filt.info["sfreq"])
    events, event_id = mne.events_from_annotations(raw_filt, verbose=False)
    
    t1_code = event_id.get("T1", -1)
    t2_code = event_id.get("T2", -1)
    task_events = [ev for ev in events if ev[2] in [t1_code, t2_code]]

    retained_epochs = []
    retained_labels = []
    trial_audit_rows = []

    filt_data = raw_filt.get_data() # shape: (3, n_times) in Volts

    for tr_idx, ev in enumerate(task_events):
        onset_sample = int(ev[0])
        onset_sec = onset_sample / sfreq
        duration_sec = EPOCH_DURATION_SEC
        end_sec = onset_sec + duration_sec
        n_samples = int(duration_sec * sfreq) # 640
        end_sample = onset_sample + n_samples

        cue_code = "T1" if ev[2] == t1_code else "T2"
        class_label = LABEL_LEFT_FIST if cue_code == "T1" else LABEL_RIGHT_FIST
        label_name = LABEL_NAMES[class_label]

        # Check overlap with zero blocks
        has_zero_overlap = False
        overlap_details = ""
        for (zs, ze, zd, zcount) in zero_intervals:
            if max(onset_sec, zs) < min(end_sec, ze):
                has_zero_overlap = True
                overlap_sec = min(end_sec, ze) - max(onset_sec, zs)
                overlap_details = f"Overlaps zero interval [{zs:.2f}s - {ze:.2f}s] by {overlap_sec:.3f}s"
                break

        # Check bounds
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
                "max_ptp_uv": None
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
                "exclusion_reason": f"Zero-interval contamination ({overlap_details})",
                "is_finite": None,
                "is_flat": None,
                "c3_ptp_uv": None,
                "cz_ptp_uv": None,
                "c4_ptp_uv": None,
                "max_ptp_uv": None
            })
            continue

        # Extract 3-channel epoch and convert from V to µV
        epoch_uv = filt_data[:, onset_sample:end_sample] * 1e6 # shape: (3, 640)

        # Quality screening
        is_finite = bool(np.all(np.isfinite(epoch_uv)))
        is_flat = bool(np.any(np.std(epoch_uv, axis=1) < 1e-4))

        c3_ptp = float(np.ptp(epoch_uv[0]))
        cz_ptp = float(np.ptp(epoch_uv[1]))
        c4_ptp = float(np.ptp(epoch_uv[2]))
        max_ptp = max(c3_ptp, cz_ptp, c4_ptp)

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
            "c3_ptp_uv": round(c3_ptp, 2),
            "cz_ptp_uv": round(cz_ptp, 2),
            "c4_ptp_uv": round(c4_ptp, 2),
            "max_ptp_uv": round(max_ptp, 2)
        })

    return retained_epochs, retained_labels, trial_audit_rows
