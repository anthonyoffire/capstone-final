import os
import shutil
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_CSV = "dataset/filtering/duplicates.csv"
OUTPUT_DIR = "../images/real/uniform_split/mnf_dups"

# "copy" or "move"
MODE = "copy"

# Adjust as needed
NUM_WORKERS = 16

def get_relative_path(file_path):

    normalized = os.path.normpath(file_path)
    marker = "all-minus-noface"
    parts = normalized.split(os.sep)

    try:
        idx = parts.index(marker)
        return os.path.join(*parts[idx + 1 :])
    except ValueError:
        raise RuntimeError(
            f"Could not locate '{marker}' in path: {file_path}"
        )


def process_file(file_hash, src_path):
    age_dir = os.path.basename(os.path.dirname(src_path))
    filename = os.path.basename(src_path)

    dst_dir = os.path.join(
        OUTPUT_DIR,
        file_hash,
        age_dir
    )
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, filename)

    if os.path.exists(dst_path):
        stem, ext = os.path.splitext(filename)
        dst_path = os.path.join(
            dst_dir,
            f"{stem}_duplicate{ext}"
        )

    if MODE.lower() == "copy":
        shutil.copy2(src_path, dst_path)
    elif MODE.lower() == "move":
        shutil.move(src_path, dst_path)
    else:
        raise ValueError("MODE must be 'copy' or 'move'")



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    if "file_path" not in df.columns:
        raise ValueError("CSV must contain a 'file_path' column")

    rows = list(
        zip(
            df["hash"],
            df["file_path"]
        )
    )

    print(f"Found {len(rows):,} files")
    print(f"Mode: {MODE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Output: {OUTPUT_DIR}")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_file, file_hash, path): path
            for file_hash, path in rows
        }

        with tqdm(
            total=len(futures),
            desc=f"{MODE.title()}ing files",
            unit="file"
        ) as pbar:

            for future in as_completed(futures):
                try:
                    future.result()
                    success += 1
                except Exception as e:
                    failed += 1
                    print(f"\nFailed: {futures[future]}")
                    print(e)

                pbar.update(1)

    print()
    print(f"Successful: {success:,}")
    print(f"Failed: {failed:,}")


if __name__ == "__main__":
    main()