"""
MotorMind — Phase 4: Step 1
Verify raw recording properties, channel montage, annotation mapping,
and investigate the all-zero interval at the binary and sample level.
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import mne

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "dataset", "S001", "S001R04.edf")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "preprocessing")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_channel_names(raw):
    """Strip trailing dots and standardize casing to international 10-10 standard."""
    name_map = {}
    for name in raw.ch_names:
        clean = name.strip().rstrip('.').upper()
        if clean.endswith('Z'):
            clean = clean[:-1] + 'z'
        name_map[name] = clean
    raw.rename_channels(name_map)
    return raw

def inspect_raw_binary_records(edf_path):
    """Inspect raw binary bytes of the EDF file on disk to check data records 8-14."""
    print("\n--- Low-Level Binary EDF Record Inspection ---")
    with open(edf_path, "rb") as f:
        header_main = f.read(256)
        num_records = int(header_main[236:244].strip())
        record_dur = float(header_main[244:252].strip())
        ns = int(header_main[252:256].strip())
        header_bytes = int(header_main[184:192].strip())
        
        # Read signal sample counts
        f.seek(256 + ns * 16 + ns * 80 + ns * 8 + ns * 8 + ns * 8 + ns * 8 + ns * 8 + ns * 80)
        samples_per_rec = [int(f.read(8).strip()) for _ in range(ns)]
        bytes_per_rec = sum(s * 2 for s in samples_per_rec)
        
        print(f"Header size: {header_bytes} bytes | Records: {num_records} | Record duration: {record_dur}s")
        print(f"Bytes per data record: {bytes_per_rec} bytes")
        
        # Probe records 7, 8, 9, 14, 15
        for rec_idx in [7, 8, 9, 14, 15]:
            f.seek(header_bytes + rec_idx * bytes_per_rec)
            # 64 channels * 160 samples * 2 bytes = 20,480 bytes
            eeg_bytes = f.read(64 * 160 * 2)
            is_zero = all(b == 0 for b in eeg_bytes)
            print(f"  Record {rec_idx:2d} (t={rec_idx*record_dur:4.1f}s to {(rec_idx+1)*record_dur:4.1f}s): "
                  f"All 64 EEG channels exactly zero = {is_zero}")

def main():
    print("=" * 70)
    print("MotorMind Step 1: Raw Recording & Data Quality Verification")
    print("=" * 70)
    
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Missing file: {DATASET_FILE}")
        
    # 1. Direct Binary Verification
    inspect_raw_binary_records(DATASET_FILE)
    
    # 2. Load via MNE
    raw = mne.io.read_raw_edf(DATASET_FILE, preload=True, verbose=False)
    raw = sanitize_channel_names(raw)
    
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    duration = n_samples / sfreq
    n_channels = len(raw.ch_names)
    
    print("\n--- Basic Recording Metadata ---")
    print(f"Sampling Frequency (sfreq) : {sfreq} Hz")
    print(f"Total Samples              : {n_samples}")
    print(f"Duration                   : {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Total Channels             : {n_channels}")
    print(f"Motor Channels Present     : C3={'C3' in raw.ch_names}, Cz={'Cz' in raw.ch_names}, C4={'C4' in raw.ch_names}")
    
    # 3. Annotations & Events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    id_to_label = {v: k for k, v in event_id.items()}
    
    print("\n--- Event Protocol Verification (Run 4: Motor Imagery) ---")
    print(f"Event ID mapping: {event_id}")
    event_counts = {lbl: 0 for lbl in event_id.keys()}
    for ev in events:
        event_counts[id_to_label[ev[2]]] += 1
        
    for lbl, count in sorted(event_counts.items()):
        if lbl == "T0":
            desc = "Rest / Fixation interval"
        elif lbl == "T1":
            desc = "IMAGINED Left Fist (Class 0)"
        elif lbl == "T2":
            desc = "IMAGINED Right Fist (Class 1)"
        else:
            desc = "Unknown"
        print(f"  - '{lbl}': {count:2d} occurrences -> {desc}")
        
    # 4. Zero Interval Analysis
    data = raw.get_data() # (64, 20000) in Volts
    all_zeros_mask = np.all(data == 0, axis=0)
    zero_indices = np.where(all_zeros_mask)[0]
    
    print(f"\n--- Zero-Interval Scan across all {n_channels} Channels ---")
    print(f"Total sample columns with exact 0.0 V across all channels: {len(zero_indices)}")
    
    diffs = np.diff(zero_indices)
    breaks = np.where(diffs > 1)[0]
    starts = [zero_indices[0]] + [zero_indices[b + 1] for b in breaks]
    ends = [zero_indices[b] for b in breaks] + [zero_indices[-1]]
    
    for idx, (s, e) in enumerate(zip(starts, ends)):
        t_s = s / sfreq
        t_e = (e + 1) / sfreq
        print(f"  Block {idx+1}: Samples [{s}:{e+1}] -> Time [{t_s:.3f}s to {t_e:.3f}s] "
              f"(Duration: {t_e - t_s:.3f}s, {e - s + 1} samples)")
              
    # Check overlap with annotations
    print("\n--- Trial Overlap with Zero Interval ---")
    for annot_idx, a in enumerate(raw.annotations):
        a_start = a['onset']
        a_end = a['onset'] + a['duration']
        # Check against first zero block
        z_start = starts[0] / sfreq
        z_end = (ends[0] + 1) / sfreq
        if max(a_start, z_start) < min(a_end, z_end):
            print(f"  [OVERLAP DETECTED] Annotation {annot_idx} ({a['description']}): "
                  f"{a_start:.2f}s - {a_end:.2f}s overlaps zero block [{z_start:.2f}s - {z_end:.2f}s]")
            overlap_dur = min(a_end, z_end) - max(a_start, z_start)
            print(f"    -> Corrupted duration inside trial: {overlap_dur:.3f}s ({int(overlap_dur*sfreq)} samples)")

    # 5. Visualizations
    time_vec = np.arange(n_samples) / sfreq
    target_chs = ['C3', 'Cz', 'C4']
    ch_indices = [raw.ch_names.index(ch) for ch in target_chs]
    data_uv = data[ch_indices] * 1e6 # Convert V to uV
    
    # FIGURE 1: Full timeline and Zoomed Zero Interval
    fig1, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(14, 9), dpi=300)
    
    # Panel 1: Full Timeline
    colors = {'C3': '#1f77b4', 'Cz': '#2ca02c', 'C4': '#d62728'}
    for i, ch_name in enumerate(target_chs):
        ax_full.plot(time_vec, data_uv[i] + (1 - i) * 80, label=f"{ch_name} (offset {(1 - i) * 80} µV)", 
                     color=colors[ch_name], lw=0.7, alpha=0.85)
                     
    # Shade events on full timeline
    for ev in events:
        t_on = ev[0] / sfreq
        lbl = id_to_label[ev[2]]
        if lbl == 'T1':
            ax_full.axvspan(t_on, t_on + 4.0, color='red', alpha=0.15, label='T1 (Left Fist)' if t_on < 25 else "")
        elif lbl == 'T2':
            ax_full.axvspan(t_on, t_on + 4.0, color='green', alpha=0.15, label='T2 (Right Fist)' if t_on < 10 else "")
            
    # Highlight zero blocks
    ax_full.axvspan(8.0, 15.0, color='gold', alpha=0.45, hatch='//', label='All-Zero Discontinuity (8.0s - 15.0s)')
    ax_full.axvspan(124.5, 125.0, color='gray', alpha=0.45, label='End Padding Zero')
    
    ax_full.set_title("MotorMind Step 1: S001R04 Raw EEG Timeline (C3, Cz, C4) & Event Annotations", 
                      fontsize=12, fontweight='bold')
    ax_full.set_ylabel("Signal Amplitude (µV)", fontsize=10, fontweight='bold')
    ax_full.set_xlim(0, 125)
    ax_full.grid(True, linestyle='--', alpha=0.4)
    ax_full.legend(loc='upper right', ncol=3, fontsize=9)
    
    # Panel 2: Zoom on Anomaly (4s to 22s)
    zoom_mask = (time_vec >= 4.0) & (time_vec <= 22.0)
    time_zoom = time_vec[zoom_mask]
    
    for i, ch_name in enumerate(target_chs):
        ax_zoom.plot(time_zoom, data_uv[i, zoom_mask], label=f"{ch_name}", color=colors[ch_name], lw=1.1)
        
    ax_zoom.axvspan(4.2, 8.3, color='green', alpha=0.15, label="Trial 1: T2 (Right Fist Cue)")
    ax_zoom.axvspan(8.0, 15.0, color='gold', alpha=0.4, hatch='//', label="Confirmed Zero Flatline (8.0s - 15.0s)")
    ax_zoom.axvspan(16.6, 20.8, color='gray', alpha=0.15, label="T0 Rest Interval")
    ax_zoom.axvspan(20.8, 22.0, color='red', alpha=0.15, label="Trial 2: T1 (Left Fist Cue)")
    
    ax_zoom.axvline(8.0, color='darkred', linestyle='--', lw=1.5)
    ax_zoom.text(8.1, 35, "Abrupt flatline begins (t=8.0s)\nTrial 1 clipped by 0.3s", 
                color='darkred', fontsize=9, fontweight='bold')
    ax_zoom.axvline(15.0, color='darkblue', linestyle='--', lw=1.5)
    ax_zoom.text(15.1, 35, "Signal resumes (t=15.0s)", color='darkblue', fontsize=9, fontweight='bold')
    
    ax_zoom.set_title("Detailed Zoom on Anomaly: Trial 1 Truncation and 7-Second Zero Gap", 
                      fontsize=12, fontweight='bold')
    ax_zoom.set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
    ax_zoom.set_ylabel("Signal Amplitude (µV)", fontsize=10, fontweight='bold')
    ax_zoom.set_xlim(4.0, 22.0)
    ax_zoom.grid(True, linestyle='--', alpha=0.4)
    ax_zoom.legend(loc='lower right', ncol=3, fontsize=9)
    
    fig1.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "fig1_step1_recording_timeline_and_zero_interval.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"\n[Saved Figure 1] {fig1_path}")
    
    # FIGURE 2: Zoom on Clean Representative T1 and T2 Trials
    fig2, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Clean T1: Trial 2 at 20.8s (duration 4.0s)
    t1_on = 20.8
    t1_mask = (time_vec >= t1_on) & (time_vec <= t1_on + 4.0)
    t_trial = time_vec[t1_mask] - t1_on
    
    c3_idx = raw.ch_names.index('C3')
    c4_idx = raw.ch_names.index('C4')
    
    ax_t1.plot(t_trial, data_uv[0, t1_mask], label='C3 (Left Motor Cortex / Ipsilateral)', color='#1f77b4', lw=1.2)
    ax_t1.plot(t_trial, data_uv[2, t1_mask], label='C4 (Right Motor Cortex / Contralateral)', color='#d62728', lw=1.2)
    ax_t1.set_title("Representative T1 Trial: Imagined Left Fist (t=20.8s - 24.8s)", fontsize=11, fontweight='bold')
    ax_t1.set_xlabel("Trial Time (seconds)", fontsize=10, fontweight='bold')
    ax_t1.set_ylabel("Raw Amplitude (µV)", fontsize=10, fontweight='bold')
    ax_t1.grid(True, linestyle='--', alpha=0.4)
    ax_t1.legend(loc='upper right', fontsize=9)
    
    # Clean T2: Trial 3 at 29.1s (duration 4.0s)
    t2_on = 29.1
    t2_mask = (time_vec >= t2_on) & (time_vec <= t2_on + 4.0)
    t_trial2 = time_vec[t2_mask] - t2_on
    
    ax_t2.plot(t_trial2, data_uv[0, t2_mask], label='C3 (Left Motor Cortex / Contralateral)', color='#1f77b4', lw=1.2)
    ax_t2.plot(t_trial2, data_uv[2, t2_mask], label='C4 (Right Motor Cortex / Ipsilateral)', color='#d62728', lw=1.2)
    ax_t2.set_title("Representative T2 Trial: Imagined Right Fist (t=29.1s - 33.1s)", fontsize=11, fontweight='bold')
    ax_t2.set_xlabel("Trial Time (seconds)", fontsize=10, fontweight='bold')
    ax_t2.set_ylabel("Raw Amplitude (µV)", fontsize=10, fontweight='bold')
    ax_t2.grid(True, linestyle='--', alpha=0.4)
    ax_t2.legend(loc='upper right', fontsize=9)
    
    fig2.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "fig2_step1_representative_trials_T1_T2.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"[Saved Figure 2] {fig2_path}")
    print("\nStep 1 diagnostic completed successfully.")

if __name__ == "__main__":
    main()
