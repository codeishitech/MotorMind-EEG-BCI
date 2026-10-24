"""
MotorMind — Step 6A: Safe Dataset Expansion

Downloads motor-imagery runs R04, R08, and R12
for participants S001 through S040.

Reuses existing valid files, validates EDF metadata and annotations
using MNE, and saves a dataset manifest.
"""

import os
import sys
import time
import urllib.request
import urllib.error
import pandas as pd
import mne


# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "dataset_manifest.csv")

BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
PARTICIPANTS = [f"S{i:03d}" for i in range(1, 41)]
RUNS = ["R04", "R08", "R12"]

# Download settings
NETWORK_TIMEOUT = 90  # seconds
MAX_RETRIES = 5
CHUNK_SIZE = 64 * 1024  # 64 KB

def get_target_path(participant: str, run: str) -> str:
    """Returns local path: dataset/{participant}/{participant}{run}.edf"""
    return os.path.join(DATASET_DIR, participant, f"{participant}{run}.edf")

def get_remote_url(participant: str, run: str) -> str:
    """Returns remote URL: https://physionet.org/files/eegmmidb/1.0.0/{participant}/{participant}{run}.edf"""
    return f"{BASE_URL}/{participant}/{participant}{run}.edf"


def download_file(url: str, target_path: str) -> bool:
    """Download directly to the final path, avoiding .part-file renaming."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    headers = {"User-Agent": "MotorMind-Downloader/1.0"}
    started_download = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"    [Attempt {attempt}/{MAX_RETRIES}] Connecting to {url} ...")
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT) as response:
                if response.getcode() != 200:
                    raise RuntimeError(f"HTTP status {response.getcode()}")

                header_size = response.headers.get("Content-Length")
                expected_size = int(header_size) if header_size else None
                downloaded = 0
                started_download = True

                with open(target_path, "wb") as f_out:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break

                        f_out.write(chunk)
                        downloaded += len(chunk)

                        if expected_size:
                            pct = downloaded * 100 / expected_size
                            sys.stdout.write(
                                f"\r    -> Downloading: {pct:5.1f}% "
                                f"({downloaded / (1024 * 1024):.2f}/"
                                f"{expected_size / (1024 * 1024):.2f} MB)"
                            )
                            sys.stdout.flush()

            print()

            actual_size = os.path.getsize(target_path)

            if actual_size <= 100 * 1024:
                raise RuntimeError(f"File too small: {actual_size} bytes")

            if expected_size is not None and actual_size != expected_size:
                raise RuntimeError(
                    f"Incomplete download: expected {expected_size} bytes, "
                    f"received {actual_size} bytes"
                )

            print("    [Success] Download completed.")
            return True

        except KeyboardInterrupt:
            print("\n    Download interrupted.")
            raise

        except Exception as e:
            print(f"\n    [Download Error] Attempt {attempt} failed: {e}")

            # This function is called only for missing or invalid files.
            # Remove an incomplete result before retrying.
            if started_download and os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except OSError as cleanup_error:
                    print(f"    Could not remove incomplete file: {cleanup_error}")
                    return False

            started_download = False

            if attempt < MAX_RETRIES:
                wait_seconds = min(2 * attempt, 10)
                print(f"    Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    return False

def validate_edf(file_path: str):
    """
    Validates EDF recording using MNE in metadata-only mode (preload=False).
    Extracts sampling frequency, duration, channel count, and event counts.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return {
            "valid": False,
            "status": "FILE_MISSING_OR_EMPTY",
            "sfreq": None,
            "duration": None,
            "n_channels": None,
            "count_t0": 0,
            "count_t1": 0,
            "count_t2": 0,
            "event_counts_str": "None",
            "details": "File does not exist or has 0 bytes"
        }
        
    try:
        # Preload=False for metadata-only fast inspection
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        
        sfreq = float(raw.info['sfreq'])
        n_samples = raw.n_times
        duration = float(n_samples / sfreq)
        n_channels = len(raw.ch_names)
        
        # Check for core motor imagery channels
        clean_channels = [ch.strip().rstrip('.').upper() for ch in raw.ch_names]
        has_c3 = "C3" in clean_channels
        has_cz = "CZ" in clean_channels
        has_c4 = "C4" in clean_channels
        
        # Read event annotations
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        id_to_desc = {v: k for k, v in event_id.items()}
        
        counts = {}
        for ev in events:
            desc = id_to_desc.get(ev[2], "UNKNOWN")
            counts[desc] = counts.get(desc, 0) + 1
            
        count_t0 = counts.get("T0", 0)
        count_t1 = counts.get("T1", 0)
        count_t2 = counts.get("T2", 0)
        
        counts_str = ", ".join([f"{k}: {v}" for k, v in sorted(counts.items())])
        
        # Verification criteria:
        # 1. Expected sfreq around 160 Hz
        # 2. Contains core motor channels
        # 3. Contains T1 (imagined left) and T2 (imagined right) labels
        issues = []
        if sfreq != 160.0:
            issues.append(f"Unexpected sfreq ({sfreq} Hz)")
        if not (has_c3 and has_cz and has_c4):
            issues.append(f"Missing motor channels (C3:{has_c3}, Cz:{has_cz}, C4:{has_c4})")
        if count_t1 == 0 or count_t2 == 0:
            issues.append(f"Missing MI events (T1:{count_t1}, T2:{count_t2})")
            
        if issues:
            status = "VALIDATION_WARNING: " + "; ".join(issues)
            valid = False
        else:
            status = "VALID"
            valid = True
            
        return {
            "valid": valid,
            "status": status,
            "sfreq": sfreq,
            "duration": round(duration, 3),
            "n_channels": n_channels,
            "count_t0": count_t0,
            "count_t1": count_t1,
            "count_t2": count_t2,
            "event_counts_str": counts_str,
            "details": f"{n_channels} chs, {duration:.1f}s, T1={count_t1}, T2={count_t2}"
        }
    except Exception as e:
        return {
            "valid": False,
            "status": f"READ_ERROR: {str(e)}",
            "sfreq": None,
            "duration": None,
            "n_channels": None,
            "count_t0": 0,
            "count_t1": 0,
            "count_t2": 0,
            "event_counts_str": "Error",
            "details": f"MNE parse error: {str(e)}"
        }

def main():
    print("=" * 80)
    print("MotorMind — Step 6A: Safe Dataset Expansion (S001–S040, Runs R04, R08, R12)")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_expected = len(PARTICIPANTS) * len(RUNS)
    print(f"Target Scope: {len(PARTICIPANTS)} participants x {len(RUNS)} runs = {total_expected} recordings")
    print(f"Target Runs : {RUNS} (Task 2: Imagined Left Fist vs. Imagined Right Fist)")
    print(f"Target Dir  : {DATASET_DIR}\n")
    
    reused_count = 0
    downloaded_count = 0
    failed_count = 0
    valid_count = 0
    warning_or_invalid_count = 0
    
    manifest_rows = []
    
    for p in PARTICIPANTS:
        print(f"--- Processing Participant {p} ---")
        p_dir = os.path.join(DATASET_DIR, p)
        os.makedirs(p_dir, exist_ok=True)
        
        for r in RUNS:
            target_path = get_target_path(p, r)
            rel_path = os.path.relpath(target_path, BASE_DIR).replace("\\", "/")
            url = get_remote_url(p, r)
            file_name = f"{p}{r}.edf"
            
            # Step 1: Check existing file
            download_status = "UNKNOWN"
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                # Fast check if it's already valid
                pre_check = validate_edf(target_path)
                if pre_check["valid"]:
                    print(f"  [{file_name}] Reusing existing valid recording ({os.path.getsize(target_path)/(1024*1024):.2f} MB)")
                    download_status = "REUSED"
                    reused_count += 1
                else:
                    print(f"  [{file_name}] Existing file is invalid or incomplete. Re-downloading...")
                    if download_file(url, target_path):
                        download_status = "DOWNLOADED"
                        downloaded_count += 1
                    else:
                        download_status = "FAILED"
                        failed_count += 1
            else:
                print(f"  [{file_name}] Missing locally. Downloading from PhysioNet...")
                if download_file(url, target_path):
                    download_status = "DOWNLOADED"
                    downloaded_count += 1
                    print(f"    [Success] Downloaded {file_name} ({os.path.getsize(target_path)/(1024*1024):.2f} MB)")
                else:
                    download_status = "FAILED"
                    failed_count += 1
                    print(f"    [Failure] Could not download {file_name}")

            # Step 2: Validate via MNE metadata mode
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                file_size = os.path.getsize(target_path)
                val_res = validate_edf(target_path)
            else:
                file_size = 0
                val_res = {
                    "valid": False,
                    "status": "FILE_NOT_FOUND",
                    "sfreq": None,
                    "duration": None,
                    "n_channels": None,
                    "count_t0": 0,
                    "count_t1": 0,
                    "count_t2": 0,
                    "event_counts_str": "None",
                    "details": "File missing after download stage"
                }

            if val_res["valid"]:
                valid_count += 1
                status_symbol = "[OK]"
            else:
                warning_or_invalid_count += 1
                status_symbol = "[WARN/FAIL]"

            print(f"    {status_symbol} Validation: {val_res['status']} | {val_res['details']}")

            manifest_rows.append({
                "participant_id": p,
                "run": r,
                "filename": file_name,
                "file_path": rel_path,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size else 0.0,
                "download_status": download_status,
                "validation_status": val_res["status"],
                "sampling_frequency_hz": val_res["sfreq"],
                "duration_seconds": val_res["duration"],
                "n_channels": val_res["n_channels"],
                "count_t0": val_res["count_t0"],
                "count_t1": val_res["count_t1"],
                "count_t2": val_res["count_t2"],
                "all_event_counts": val_res["event_counts_str"]
            })
        print()

    # Step 3: Save Manifest CSV
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(MANIFEST_PATH, index=False)
    print("=" * 80)
    print("DOWNLOAD & VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Target Recordings : {total_expected}")
    print(f"Reused Existing Files   : {reused_count}")
    print(f"Newly Downloaded Files  : {downloaded_count}")
    print(f"Failed Downloads        : {failed_count}")
    print(f"Validated Successfully  : {valid_count}")
    print(f"Validation Issues       : {warning_or_invalid_count}")
    print(f"\nManifest saved to: {MANIFEST_PATH}")
    print("=" * 80)
    
    # Print concise manifest overview table
    summary_cols = ["participant_id", "run", "download_status", "duration_seconds", "count_t0", "count_t1", "count_t2", "validation_status"]
    print("\nDataset Manifest Table Preview:")
    print(df_manifest[summary_cols].to_string(index=False))

if __name__ == "__main__":
    main()
