"""
MotorMind — Step 4: Extract and Label EEG Trials (Epochs)
=========================================================
Extracts 4.0-second motor-imagery epochs for Left Fist (T1) and Right Fist (T2)
from continuous 8–30 Hz filtered EEG (C3, Cz, C4).

Excludes Trial 1 (T2 at t=4.2s) due to the zero-data interval contamination
verified in Step 2.

Outputs:
  - output/epochs/eeg_motor_imagery_epochs.npz: Saved trial tensor X, labels y, and metadata.
  - output/epochs/fig6_example_trials_left_vs_right.png: Beginner-friendly visualization (300 DPI).
  - learning_notes.md: Conceptual summary of epochs and trial extraction.
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# Project directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "dataset", "S001", "S001R04.edf")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "epochs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_channel_names(raw):
    """Standardize channel names to international 10-10 format."""
    clean_map = {}
    for name in raw.ch_names:
        clean = name.strip().rstrip('.').upper()
        if clean.endswith('Z'):
            clean = clean[:-1] + 'z'
        clean_map[name] = clean
    raw.rename_channels(clean_map)
    return raw

def main():
    print("=" * 70)
    print("MotorMind Step 4: Extract and Label EEG Trials (Epochs)")
    print("=" * 70)

    # 1. Load Raw EDF (Original file remains unchanged on disk)
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_FILE}")

    print(f"Loading raw recording: {os.path.basename(DATASET_FILE)}...")
    raw = mne.io.read_raw_edf(DATASET_FILE, preload=True, verbose=False)
    raw = sanitize_channel_names(raw)

    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    print(f"Sampling rate: {sfreq} Hz | Total duration: {n_samples / sfreq:.2f} s")

    # 2. Apply 8–30 Hz Band-Pass Filter (Identical to Step 3)
    print("\nApplying 8–30 Hz zero-phase FIR band-pass filter...")
    raw_filt = raw.copy().filter(
        l_freq=8.0,
        h_freq=30.0,
        fir_design='firwin',
        phase='zero',
        verbose=False
    )

    # 3. Channel Selection: C3, Cz, C4
    target_channels = ['C3', 'Cz', 'C4']
    print(f"Selecting target motor channels: {target_channels}")
    raw_filt.pick(target_channels)

    # 4. Extract Events and Filter for Task Cues
    events, event_id = mne.events_from_annotations(raw_filt, verbose=False)
    print(f"Event ID mapping: {event_id}")

    t1_code = event_id['T1'] # Imagined Left Fist
    t2_code = event_id['T2'] # Imagined Right Fist

    task_events = [ev for ev in events if ev[2] in [t1_code, t2_code]]
    print(f"\nTotal task cues detected in recording: {len(task_events)} (7 Left Fist, 7 Right Fist)")

    # 5. Exclude Damaged Trial 1
    # Trial 1 is the first T2 event (onset t = 4.20s, sample 672)
    first_t2_idx = next(i for i, ev in enumerate(task_events) if ev[2] == t2_code)
    excluded_event = task_events[first_t2_idx]
    excluded_time = excluded_event[0] / sfreq

    print("\n--- Trial Exclusion ---")
    print(f"Excluded Trial: Trial 1 (T2 at t = {excluded_time:.2f}s, sample {excluded_event[0]})")
    print("Reason: Overlaps the 8.0s–15.0s all-channel zero flatline by 0.30s (documented in Step 2).")

    clean_task_events = [ev for i, ev in enumerate(task_events) if i != first_t2_idx]
    clean_task_events = np.array(clean_task_events)

    n_left = sum(1 for ev in clean_task_events if ev[2] == t1_code)
    n_right = sum(1 for ev in clean_task_events if ev[2] == t2_code)
    print(f"Retained Clean Trials: {len(clean_task_events)} total ({n_left} Left Fist, {n_right} Right Fist)")

    # 6. Extract 4.0-Second Epochs
    # tmin = 0.0s, tmax = 4.0s - 1/sfreq (exact 640 samples at 160 Hz)
    epoch_duration_sec = 4.0
    tmax = epoch_duration_sec - (1.0 / sfreq)
    
    epochs = mne.Epochs(
        raw_filt,
        clean_task_events,
        event_id={'T1_Left': t1_code, 'T2_Right': t2_code},
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    # 7. Convert to Numerical Arrays for Machine Learning
    # Convert from Volts to Microvolts (µV)
    X = epochs.get_data() * 1e6 # Shape: (n_epochs, n_channels, n_times)
    # Binary class labels: 0 = Left Fist (T1), 1 = Right Fist (T2)
    y = np.array([0 if ev[2] == t1_code else 1 for ev in clean_task_events], dtype=int)
    y_labels = np.array(['Left_Fist' if val == 0 else 'Right_Fist' for val in y])

    n_epochs, n_ch, n_times = X.shape
    print("\n--- Extracted Epoch Dataset Dimensions ---")
    print(f"Trial Tensor X shape : {X.shape} -> (Number of Trials: {n_epochs}, Channels: {n_ch}, Timepoints: {n_times})")
    print(f"Labels vector y shape: {y.shape} -> Binary: {list(y)}")
    print(f"Epoch duration       : {n_times / sfreq:.2f} seconds ({n_times} samples per trial)")
    print(f"Signal amplitude unit: Microvolts (µV)")

    # 8. Save Dataset in Reproducible NumPy Format
    save_file_path = os.path.join(OUTPUT_DIR, "eeg_motor_imagery_epochs.npz")
    np.savez_compressed(
        save_file_path,
        X=X,
        y=y,
        y_labels=y_labels,
        channel_names=target_channels,
        sfreq=sfreq,
        epoch_duration_sec=epoch_duration_sec,
        units="uV",
        subject="S001",
        run="R04"
    )
    print(f"\n[Saved Epochs Dataset] {save_file_path}")

    # =========================================================================
    # 9. Beginner-Friendly Visualization (Figure 6)
    # =========================================================================
    print("Generating Figure 6: Example Left-Hand vs Right-Hand Trials...")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    trial_time = np.arange(n_times) / sfreq

    # Find first clean Left-Hand trial and first clean Right-Hand trial
    left_trial_idx = next(i for i, label in enumerate(y) if label == 0) # Trial 2 in recording
    right_trial_idx = next(i for i, label in enumerate(y) if label == 1) # Trial 3 in recording

    colors = {'C3': '#1f77b4', 'Cz': '#2ca02c', 'C4': '#d62728'}

    # Panel A: Left Fist Trial
    for ch_idx, ch_name in enumerate(target_channels):
        ax_left.plot(trial_time, X[left_trial_idx, ch_idx], label=f"{ch_name}", color=colors[ch_name], lw=1.3)
    ax_left.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax_left.set_title("A. Example Left-Hand Imagery Trial (T1)\n[Onset t = 20.8s, Class Label = 0]", 
                      fontsize=11, fontweight='bold')
    ax_left.set_xlabel("Time from Visual Cue (seconds)", fontsize=10, fontweight='bold')
    ax_left.set_ylabel("Filtered Amplitude (µV)", fontsize=10, fontweight='bold')
    ax_left.set_xlim(0, 4.0)
    ax_left.set_ylim(-35, 35)
    ax_left.grid(True, linestyle='--', alpha=0.4)
    ax_left.legend(loc='upper right', fontsize=9)

    # Panel B: Right Fist Trial
    for ch_idx, ch_name in enumerate(target_channels):
        ax_right.plot(trial_time, X[right_trial_idx, ch_idx], label=f"{ch_name}", color=colors[ch_name], lw=1.3)
    ax_right.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax_right.set_title("B. Example Right-Hand Imagery Trial (T2)\n[Onset t = 29.1s, Class Label = 1]", 
                       fontsize=11, fontweight='bold')
    ax_right.set_xlabel("Time from Visual Cue (seconds)", fontsize=10, fontweight='bold')
    ax_right.set_ylabel("Filtered Amplitude (µV)", fontsize=10, fontweight='bold')
    ax_right.set_xlim(0, 4.0)
    ax_right.set_ylim(-35, 35)
    ax_right.grid(True, linestyle='--', alpha=0.4)
    ax_right.legend(loc='upper right', fontsize=9)

    fig.suptitle("MotorMind Step 4: Representative 4.0-Second Filtered EEG Trials (C3, Cz, C4)", 
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig6_example_trials_left_vs_right.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved Figure 6] {fig_path}")

    # =========================================================================
    # 10. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4 EXECUTION COMPLETED:")
    print(f"  - Extracted trials: {n_epochs} (Left: {n_left}, Right: {n_right})")
    print(f"  - Array dimensions: (13 trials, 3 channels, 640 timepoints)")
    print(f"  - Saved data file : {save_file_path}")
    print(f"  - Saved plot      : {fig_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
