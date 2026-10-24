import os
import urllib.request
import sys

BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0/S001/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "S001")

FILES = [
    "S001R03.edf",
    "S001R03.edf.event",
    "S001R04.edf",
    "S001R04.edf.event"
]

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\r  -> Progress: {percent:5.1f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
        sys.stdout.flush()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Target directory: {OUTPUT_DIR}")
    
    for filename in FILES:
        target_path = os.path.join(OUTPUT_DIR, filename)
        url = BASE_URL + filename
        if os.path.exists(target_path):
            print(f"File already exists: {filename} ({os.path.getsize(target_path)} bytes)")
            continue
        print(f"Downloading {filename} from {url}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', -1))
                block_size = 1024 * 64
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        percent = min(100, downloaded * 100 / total_size)
                        sys.stdout.write(f"\r  -> Progress: {percent:5.1f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                        sys.stdout.flush()
                print(f"\n  [Done] Saved {filename} ({os.path.getsize(target_path)} bytes)")
        except Exception as e:
            print(f"\n  [Error] Failed to download {filename}: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)
            raise e

if __name__ == "__main__":
    main()
