import csv
import hashlib
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

ROOT_DIRS = [
    r"../images/src/imdb_crop",
]

OUTPUT_DIR = r"../images/src/"

HASH_CHUNK_SIZE = 1024 * 1024
MAX_WORKERS = min(32, (os.cpu_count() or 1) * 4)


def scan_files(root):
    files = []

    def recurse(path):
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            files.append(entry.path)
                        elif entry.is_dir(follow_symlinks=False):
                            recurse(entry.path)
                    except PermissionError:
                        pass
        except PermissionError:
            pass

    recurse(root)
    return files


def sha1_file(filepath):
    h = hashlib.sha1()

    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(HASH_CHUNK_SIZE):
                h.update(chunk)

        return filepath, h.hexdigest()

    except Exception as e:
        print(f"Failed: {filepath} ({e})")
        return None


def duplicate_count(hashes):
    counts = Counter(hashes)
    return sum(count for count in counts.values() if count > 1)


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_hashes = []

    for root in ROOT_DIRS:
        root = os.path.abspath(root)
        name = os.path.basename(root.rstrip("\\/"))

        print(f"\n[{name}] Scanning...")

        files = scan_files(root)

        print(f"[{name}] Found {len(files):,} files")
        print(f"[{name}] Hashing...")

        root_hashes = []
        hash_to_files = {}

        all_csv_path = output_dir / f"{name}_sha1.csv"
        dup_csv_path = output_dir / f"{name}_duplicates.csv"

        with open(all_csv_path, "w", newline="", encoding="utf-8") as all_f:
            all_writer = csv.writer(all_f)
            all_writer.writerow(["hash", "file_path"])

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = executor.map(sha1_file, files, chunksize=32)

                for result in tqdm(results, total=len(files), desc=name, unit="file"):
                    if result is None:
                        continue

                    filepath, sha1 = result

                    full_path = os.path.abspath(filepath).replace("\\", "/")

                    all_writer.writerow([sha1, full_path])

                    root_hashes.append(sha1)
                    all_hashes.append(sha1)

                    hash_to_files.setdefault(sha1, []).append(full_path)

        with open(dup_csv_path, "w", newline="", encoding="utf-8") as dup_f:
            dup_writer = csv.writer(dup_f)
            dup_writer.writerow(["hash", "file_path"])

            for sha1, file_list in hash_to_files.items():
                if len(file_list) > 1:
                    for file_path in file_list:
                        dup_writer.writerow([sha1, file_path])

        duplicate_files = duplicate_file_count(root_hashes)
        duplicate_groups = duplicate_group_count(root_hashes)

        print(f"[{name}] Files participating in duplicates: {duplicate_files:,}")
        print(f"[{name}] Duplicate groups: {duplicate_groups:,}")
        print(f"[{name}] All files CSV: {all_csv_path}")
        print(f"[{name}] Duplicate files CSV: {dup_csv_path}")

    total_duplicate_files = duplicate_file_count(all_hashes)
    total_duplicate_groups = duplicate_group_count(all_hashes)

    print("\n========== SUMMARY ==========")
    print(f"Files participating in duplicates across all roots: {total_duplicate_files:,}")
    print(f"Duplicate groups across all roots: {total_duplicate_groups:,}")

def duplicate_file_count(hashes):
    counts = Counter(hashes)
    return sum(count for count in counts.values() if count > 1)


def duplicate_group_count(hashes):
    counts = Counter(hashes)
    return sum(1 for count in counts.values() if count > 1)

if __name__ == "__main__":
    main()