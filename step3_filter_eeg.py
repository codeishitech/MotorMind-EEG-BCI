"""
MotorMind — Step 3: Implement Basic EEG Filtering
=================================================
Applies an 8–30 Hz zero-phase FIR band-pass filter to the continuous EEG data
of S001R04.edf, leaving the original recording intact.

Visualizations:
  - fig4_raw_vs_filtered.png: Raw vs. Filtered C3 signal over t = 20.0s to 25.0s (300 DPI)
  - fig5_filter_psd_comparison.png: Power Spectral Density before and after filtering for C3, Cz, C4 (300 DPI)
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# Define directories reusing existing project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "dataset", "S001", "S001R04.edf")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "preprocessing")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_channel_names(raw):
    """
    Standardizes channel names from PhysioNet format (e.g., 'C3..', 'Cz..')
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

def main():
    print("=" * 70)
    print("MotorMind Step 3: Basic EEG Filtering (8–30 Hz FIR Band-Pass)")
    print("=" * 70)

    # 1. Load Raw EDF (Original file remains unchanged on disk)
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_FILE}")

    print(f"Loading raw recording: {os.path.basename(DATASET_FILE)}...")
    raw = mne.io.read_raw_edf(DATASET_FILE, preload=True, verbose=False)
    raw = sanitize_channel_names(raw)

    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    time_vec = np.arange(n_samples) / sfreq
    print(f"Sampling frequency: {sfreq} Hz | Total duration: {n_samples / sfreq:.2f} s")

    # 2. Apply 8–30 Hz Band-Pass Filter
    # Settings explanation:
    # - l_freq = 8.0 Hz: Attenuates low-frequency drift (<0.5 Hz) and ocular artifacts (1-4 Hz delta).
    # - h_freq = 30.0 Hz: Attenuates high-frequency muscle tension (EMG >30 Hz) and 60 Hz line noise.
    # - fir_design = 'firwin': Uses MNE's standard windowed sinc FIR design with a Hamming window.
    # - phase = 'zero': Performs forward-backward filtering to achieve zero phase distortion (0 ms delay).
    print("\nApplying 8–30 Hz zero-phase FIR band-pass filter on in-memory copy...")
    raw_filtered = raw.copy().filter(
        l_freq=8.0,
        h_freq=30.0,
        fir_design='firwin',
        phase='zero',
        verbose=False
    )

    # Target motor strip channels
    target_channels = ['C3', 'Cz', 'C4']
    ch_indices = [raw.ch_names.index(ch) for ch in target_channels]
    c3_idx = raw.ch_names.index('C3')

    # Convert data from Volts (V) to Microvolts (µV)
    data_raw_uv = raw.get_data() * 1e6
    data_filt_uv = raw_filtered.get_data() * 1e6

    # =========================================================================
    # 3. Figure 4: Raw vs. Filtered C3 Signal (Clean Interval: 20s to 25s)
    # =========================================================================
    print("\nGenerating Figure 4: Raw vs Filtered C3 signal (20s to 25s)...")
    interval_mask = (time_vec >= 20.0) & (time_vec <= 25.0)
    t_interval = time_vec[interval_mask]
    
    c3_raw_segment = data_raw_uv[c3_idx, interval_mask]
    c3_filt_segment = data_filt_uv[c3_idx, interval_mask]

    fig4, (ax_overlay, ax_stacked) = plt.subplots(2, 1, figsize=(13, 8), dpi=300)

    # Subplot 1: Direct Overlay
    ax_overlay.plot(t_interval, c3_raw_segment, color='#7f7f7f', lw=1.2, alpha=0.8, label='Raw C3 (with slow baseline drift)')
    ax_overlay.plot(t_interval, c3_filt_segment, color='#1f77b4', lw=1.5, label='Filtered C3 (8–30 Hz zero-phase FIR)')
    ax_overlay.axvspan(20.8, 24.9, color='#d95f02', alpha=0.15, label='Trial 2 Active Window (T1 Left Fist)')
    ax_overlay.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax_overlay.set_title("A. Superimposed Comparison: Raw vs. 8–30 Hz Filtered C3", fontsize=11, fontweight='bold')
    ax_overlay.set_ylabel("Amplitude (µV)", fontsize=10, fontweight='bold')
    ax_overlay.set_xlim(20.0, 25.0)
    ax_overlay.grid(True, linestyle='--', alpha=0.4)
    ax_overlay.legend(loc='upper right', fontsize=9)

    # Subplot 2: Stacked View for Unobstructed Waveform Comparison
    offset = 40.0 # vertical shift in uV for clarity
    ax_stacked.plot(t_interval, c3_raw_segment + offset, color='#7f7f7f', lw=1.2, label=f'Raw C3 (+{offset:.0f} µV offset)')
    ax_stacked.plot(t_interval, c3_filt_segment - offset, color='#1f77b4', lw=1.3, label=f'Filtered C3 (-{offset:.0f} µV offset)')
    ax_stacked.axvspan(20.8, 24.9, color='#d95f02', alpha=0.12, label='Trial 2 Active Window')
    ax_stacked.axhline(offset, color='gray', linestyle=':', alpha=0.4)
    ax_stacked.axhline(-offset, color='gray', linestyle=':', alpha=0.4)
    ax_stacked.set_title("B. Stacked View: DC Drift Removal & SMR Waveform Isolation", fontsize=11, fontweight='bold')
    ax_stacked.set_xlabel("Time from Recording Start (seconds)", fontsize=10, fontweight='bold')
    ax_stacked.set_ylabel("Amplitude (µV)", fontsize=10, fontweight='bold')
    ax_stacked.set_xlim(20.0, 25.0)
    ax_stacked.grid(True, linestyle='--', alpha=0.4)
    ax_stacked.legend(loc='upper right', fontsize=9)

    fig4.suptitle("MotorMind Step 3: Time-Domain Filter Effects on Electrode C3 (Clean Interval: 20s–25s)", 
                  fontsize=12, fontweight='bold', y=0.99)
    fig4.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR, "fig4_raw_vs_filtered.png")
    fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"  [Saved] {fig4_path}")

    # =========================================================================
    # 4. Figure 5: Power Spectral Density (PSD) Comparison (C3, Cz, C4)
    # =========================================================================
    print("\nGenerating Figure 5: Power Spectral Density comparison for C3, Cz, C4...")
    fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # Compute PSDs across the entire recording (1 to 50 Hz)
    psd_raw = raw.compute_psd(fmin=1.0, fmax=50.0, picks=target_channels, verbose=False)
    psd_filt = raw_filtered.compute_psd(fmin=1.0, fmax=50.0, picks=target_channels, verbose=False)

    freqs = psd_raw.freqs
    # Convert V^2/Hz to dB relative to 1 µV^2/Hz
    psd_raw_db = 10 * np.log10(psd_raw.get_data() * 1e12)
    psd_filt_db = 10 * np.log10(psd_filt.get_data() * 1e12)

    colors = {'C3': '#1f77b4', 'Cz': '#2ca02c', 'C4': '#d62728'}

    for idx, ch_name in enumerate(target_channels):
        ax = axes5[idx]
        ax.plot(freqs, psd_raw_db[idx], color='#7f7f7f', lw=1.5, label='Raw (Unfiltered)')
        ax.plot(freqs, psd_filt_db[idx], color=colors[ch_name], lw=1.8, label=f'8–30 Hz Filtered ({ch_name})')

        # Highlight sensorimotor bands within passband
        ax.axvspan(8.0, 12.0, color='gold', alpha=0.25, label='Mu band (8–12 Hz)' if idx == 0 else "")
        ax.axvspan(13.0, 30.0, color='lightgreen', alpha=0.25, label='Beta band (13–30 Hz)' if idx == 0 else "")

        # Mark cutoff frequencies
        ax.axvline(8.0, color='red', linestyle='--', alpha=0.7, lw=1.1, label='Cutoffs (8 & 30 Hz)' if idx == 0 else "")
        ax.axvline(30.0, color='red', linestyle='--', alpha=0.7, lw=1.1)

        ax.set_title(f"Electrode {ch_name} Power Spectrum", fontsize=11, fontweight='bold')
        ax.set_xlabel("Frequency (Hz)", fontsize=10, fontweight='bold')
        ax.set_ylabel("PSD (dB / µV²/Hz)", fontsize=10, fontweight='bold')
        ax.set_xlim(1.0, 50.0)
        ax.set_ylim(-15, 45)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=8.5)

    fig5.suptitle("MotorMind Step 3: Spectral Attenuation Verification (C3, Cz, C4)", 
                  fontsize=13, fontweight='bold', y=1.02)
    fig5.tight_layout()
    fig5_path = os.path.join(OUTPUT_DIR, "fig5_filter_psd_comparison.png")
    fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print(f"  [Saved] {fig5_path}")

    # =========================================================================
    # 5. Summary Output
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3 FILTERING SUMMARY:")
    print("=" * 70)
    print(f"Filter type            : Windowed sinc FIR (fir_design='firwin', Hamming window)")
    print(f"Cutoff frequencies     : 8.0 Hz (low) to 30.0 Hz (high)")
    print(f"Phase behavior         : Zero-phase forward-backward filtering (phase='zero')")
    print(f"Sampling frequency     : {sfreq} Hz")
    print(f"Channels processed     : All 64 channels in memory (original file untouched)")
    print(f"Target motor channels  : {target_channels}")
    print(f"Output Figure 4        : {fig4_path}")
    print(f"Output Figure 5        : {fig5_path}")
    print("Execution completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
