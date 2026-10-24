import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset", "S001")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_channel_names(raw):
    """
    PhysioNet BCI2000 EDF files frequently include trailing dots in channel names
    e.g. 'Fc5.' or 'C3..'. This cleans them and standardizes to 10-10 format.
    """
    clean_names = {}
    for name in raw.ch_names:
        clean = name.strip().rstrip('.').upper()
        # Some special cases if needed:
        if clean.endswith('Z'):
            clean = clean[:-1] + 'z'
        clean_names[name] = clean
    raw.rename_channels(clean_names)
    return raw

def explore_run(file_path, run_label, is_imagined=True):
    print("=" * 70)
    print(f"LOADING FILE: {os.path.basename(file_path)} ({run_label})")
    print("=" * 70)
    
    # 1. Load Raw EDF
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw = sanitize_channel_names(raw)
    
    # 2. Inspect Metadata
    sfreq = raw.info['sfreq']
    n_channels = len(raw.ch_names)
    duration_sec = raw.n_times / sfreq
    n_samples = raw.n_times
    
    print(f"Sampling Frequency (sfreq) : {sfreq} Hz")
    print(f"Total Time Points (samples): {n_samples}")
    print(f"Recording Duration         : {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
    print(f"Number of Channels         : {n_channels}")
    print(f"Channel Names (first 10)   : {raw.ch_names[:10]}")
    print(f"Sensorimotor Channels      : {[ch for ch in ['C3', 'CZ', 'Cz', 'C4'] if ch in raw.ch_names]}")
    print(f"Highpass filter in file    : {raw.info['highpass']} Hz")
    print(f"Lowpass filter in file     : {raw.info['lowpass']} Hz")
    
    # 3. Annotations & Events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print("\nAnnotations found in file:")
    print(f"  Event ID mapping: {event_id}")
    
    # Invert event_id mapping: int_code -> label string
    id_to_label = {v: k for k, v in event_id.items()}
    
    event_counts = {}
    for ev in events:
        lbl = id_to_label[ev[2]]
        event_counts[lbl] = event_counts.get(lbl, 0) + 1
        
    print("  Event Counts:")
    for lbl, cnt in sorted(event_counts.items()):
        if lbl == "T0":
            meaning = "Rest / Inter-trial Interval"
        elif lbl == "T1":
            meaning = "IMAGINED Left Fist" if is_imagined else "ACTUAL Left Fist"
        elif lbl == "T2":
            meaning = "IMAGINED Right Fist" if is_imagined else "ACTUAL Right Fist"
        else:
            meaning = "Unknown code"
        print(f"    - '{lbl}': {cnt:2d} trials -> {meaning}")
        
    return raw, events, event_id

def plot_exploration(raw_mi, events_mi, event_id_mi, raw_act, events_act, event_id_act):
    print("\nGenerating exploration visualizations...")
    sfreq = raw_mi.info['sfreq']
    id_to_label_mi = {v: k for k, v in event_id_mi.items()}
    
    # Target channels: C3 (left motor cortex), Cz (central), C4 (right motor cortex)
    target_channels = [ch for ch in ['C3', 'Cz', 'CZ', 'C4'] if ch in raw_mi.ch_names]
    # Standardize casing to find C3, Cz, C4
    c3_ch = next((c for c in raw_mi.ch_names if c.upper() == 'C3'), None)
    cz_ch = next((c for c in raw_mi.ch_names if c.upper() in ['CZ', 'CZ..']), None)
    c4_ch = next((c for c in raw_mi.ch_names if c.upper() == 'C4'), None)
    
    selected_channels = [c for c in [c3_ch, cz_ch, c4_ch] if c is not None]
    
    # FIGURE 1: Timeline of Events & Multi-Channel Raw Signal
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    time_vec = np.arange(raw_mi.n_times) / sfreq
    
    data_mi, _ = raw_mi[selected_channels, :]
    # Convert V to uV
    data_mi_uv = data_mi * 1e6
    
    for idx, ch_name in enumerate(selected_channels):
        axes[idx].plot(time_vec, data_mi_uv[idx], color='#1f77b4', lw=0.8, alpha=0.9, label=f'{ch_name} (µV)')
        axes[idx].set_ylabel(f"{ch_name} (µV)", fontsize=11, fontweight='bold')
        axes[idx].grid(True, linestyle='--', alpha=0.4)
        axes[idx].legend(loc='upper right')
        
        # Shade event intervals
        for ev in events_mi:
            t_onset = ev[0] / sfreq
            lbl = id_to_label_mi[ev[2]]
            if lbl == 'T1': # Imagined Left
                axes[idx].axvspan(t_onset, t_onset + 4.0, color='crimson', alpha=0.18, label='T1 (Left)' if (idx==0 and t_onset < 20) else None)
            elif lbl == 'T2': # Imagined Right
                axes[idx].axvspan(t_onset, t_onset + 4.0, color='forestgreen', alpha=0.18, label='T2 (Right)' if (idx==0 and t_onset < 20) else None)
            elif lbl == 'T0': # Rest
                axes[idx].axvline(t_onset, color='gray', linestyle=':', alpha=0.5)
                
    axes[0].set_title("MotorMind: S001R04 (Imagined Movement) — Motor Strip Electrodes & Task Events", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Time (seconds)", fontsize=11, fontweight='bold')
    axes[2].set_xlim(0, min(60, time_vec[-1])) # Show first 60 seconds for clear inspection
    fig.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "1_raw_signal_and_events_S001R04.png")
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {fig1_path}")
    
    # FIGURE 2: Zoomed-in Single Trial (Imagined Left Fist: C3 vs C4)
    # Find first T1 event
    t1_events = [ev for ev in events_mi if id_to_label_mi[ev[2]] == 'T1']
    if t1_events:
        first_t1 = t1_events[0]
        start_samp = int(first_t1[0])
        end_samp = int(start_samp + 4.0 * sfreq)
        trial_time = np.arange(end_samp - start_samp) / sfreq
        
        trial_data_uv = raw_mi[[c3_ch, c4_ch], start_samp:end_samp][0] * 1e6
        
        fig2, ax = plt.subplots(figsize=(12, 5))
        ax.plot(trial_time, trial_data_uv[0], label=f'C3 (Left hemisphere / Ipsilateral to left hand)', color='#2b5c8f', lw=1.2)
        ax.plot(trial_time, trial_data_uv[1], label=f'C4 (Right hemisphere / Contralateral to left hand)', color='#e05638', lw=1.2)
        ax.set_title("MotorMind: Zoomed Single Trial (4.0s Imagined Left Fist Cue 'T1')", fontsize=13, fontweight='bold')
        ax.set_xlabel("Trial Time (seconds)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Amplitude (µV)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', frameon=True)
        fig2.tight_layout()
        fig2_path = os.path.join(OUTPUT_DIR, "2_zoomed_single_trial_T1.png")
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)
        print(f"  [Saved] {fig2_path}")
        
    # FIGURE 3: Comparison between Actual Movement (R03) and Imagined Movement (R04)
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8))
    
    # Signal variance comparison across channels
    data_act_uv = raw_act.get_data() * 1e6
    data_mi_uv_all = raw_mi.get_data() * 1e6
    
    var_act = np.std(data_act_uv, axis=1)
    var_mi = np.std(data_mi_uv_all, axis=1)
    
    # Bar comparison for motor channels
    ch_indices = [raw_mi.ch_names.index(c) for c in selected_channels]
    x = np.arange(len(selected_channels))
    width = 0.35
    
    axes3[0, 0].bar(x - width/2, [var_act[i] for i in ch_indices], width, label='Actual Movement (R03)', color='#3366cc')
    axes3[0, 0].bar(x + width/2, [var_mi[i] for i in ch_indices], width, label='Imagined Movement (R04)', color='#dc3912')
    axes3[0, 0].set_xticks(x)
    axes3[0, 0].set_xticklabels(selected_channels)
    axes3[0, 0].set_ylabel("Std Dev Amplitude (µV)", fontweight='bold')
    axes3[0, 0].set_title("Signal Amplitude / Variability Comparison", fontweight='bold')
    axes3[0, 0].legend()
    axes3[0, 0].grid(True, linestyle='--', alpha=0.4)
    
    # Event distributions
    id_to_lbl_act = {v: k for k, v in event_id_act.items()}
    act_counts = [sum(1 for ev in events_act if id_to_lbl_act[ev[2]] == k) for k in ['T0', 'T1', 'T2']]
    mi_counts = [sum(1 for ev in events_mi if id_to_label_mi[ev[2]] == k) for k in ['T0', 'T1', 'T2']]
    
    axes3[0, 1].bar(np.arange(3) - width/2, act_counts, width, label='Actual Run (R03)', color='#3366cc')
    axes3[0, 1].bar(np.arange(3) + width/2, mi_counts, width, label='Imagined Run (R04)', color='#dc3912')
    axes3[0, 1].set_xticks(np.arange(3))
    axes3[0, 1].set_xticklabels(['T0 (Rest)', 'T1 (Left)', 'T2 (Right)'])
    axes3[0, 1].set_ylabel("Trial Count", fontweight='bold')
    axes3[0, 1].set_title("Event Counts and Class Balance", fontweight='bold')
    axes3[0, 1].legend()
    axes3[0, 1].grid(True, linestyle='--', alpha=0.4)
    
    # Power Spectral Density (PSD) via MNE
    psd_act = raw_act.compute_psd(fmin=2, fmax=40, picks=selected_channels, verbose=False)
    psd_mi = raw_mi.compute_psd(fmin=2, fmax=40, picks=selected_channels, verbose=False)
    
    freqs = psd_act.freqs
    psd_act_data = psd_act.get_data() # (n_channels, n_freqs) in V^2/Hz
    psd_mi_data = psd_mi.get_data()
    
    # Plot C3 PSD
    c3_idx = selected_channels.index(c3_ch)
    axes3[1, 0].plot(freqs, 10 * np.log10(psd_act_data[c3_idx] * 1e12), label='Actual (R03)', color='#3366cc', lw=1.5)
    axes3[1, 0].plot(freqs, 10 * np.log10(psd_mi_data[c3_idx] * 1e12), label='Imagined (R04)', color='#dc3912', lw=1.5)
    axes3[1, 0].axvspan(8, 12, color='gold', alpha=0.2, label='Mu band (8-12 Hz)')
    axes3[1, 0].axvspan(13, 30, color='lightgreen', alpha=0.2, label='Beta band (13-30 Hz)')
    axes3[1, 0].set_title("Electrode C3: Power Spectral Density", fontweight='bold')
    axes3[1, 0].set_xlabel("Frequency (Hz)", fontweight='bold')
    axes3[1, 0].set_ylabel("PSD (dB / µV²/Hz)", fontweight='bold')
    axes3[1, 0].legend()
    axes3[1, 0].grid(True, linestyle='--', alpha=0.4)
    
    # Plot C4 PSD
    c4_idx = selected_channels.index(c4_ch)
    axes3[1, 1].plot(freqs, 10 * np.log10(psd_act_data[c4_idx] * 1e12), label='Actual (R03)', color='#3366cc', lw=1.5)
    axes3[1, 1].plot(freqs, 10 * np.log10(psd_mi_data[c4_idx] * 1e12), label='Imagined (R04)', color='#dc3912', lw=1.5)
    axes3[1, 1].axvspan(8, 12, color='gold', alpha=0.2, label='Mu band (8-12 Hz)')
    axes3[1, 1].axvspan(13, 30, color='lightgreen', alpha=0.2, label='Beta band (13-30 Hz)')
    axes3[1, 1].set_title("Electrode C4: Power Spectral Density", fontweight='bold')
    axes3[1, 1].set_xlabel("Frequency (Hz)", fontweight='bold')
    axes3[1, 1].set_ylabel("PSD (dB / µV²/Hz)", fontweight='bold')
    axes3[1, 1].legend()
    axes3[1, 1].grid(True, linestyle='--', alpha=0.4)
    
    fig3.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "3_actual_vs_imagined_comparison.png")
    fig3.savefig(fig3_path, dpi=150)
    plt.close(fig3)
    print(f"  [Saved] {fig3_path}")

def main():
    file_mi = os.path.join(DATASET_DIR, "S001R04.edf")
    file_act = os.path.join(DATASET_DIR, "S001R03.edf")
    
    if not os.path.exists(file_mi) or not os.path.exists(file_act):
        print(f"Error: Missing files in {DATASET_DIR}. Please run download_subset.py first.")
        return
        
    raw_mi, events_mi, event_id_mi = explore_run(file_mi, "Task 2: Motor Imagery", is_imagined=True)
    raw_act, events_act, event_id_act = explore_run(file_act, "Task 1: Motor Execution", is_imagined=False)
    
    plot_exploration(raw_mi, events_mi, event_id_mi, raw_act, events_act, event_id_act)
    print("\nExploration completed successfully! Output images saved in 'output/' directory.")

if __name__ == "__main__":
    main()
