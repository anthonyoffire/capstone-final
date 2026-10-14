import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================

SOURCE_DIR = r"../images/real/uniform_split/all"
TARGET_DIR = r"../images/real/population_split/all"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

NUM_THREADS = 16

# ==========================================================
# HASHING
# ==========================================================

def sha1_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha1()

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)

    return h.hexdigest()


def hash_file_safe(path):
    try:
        return sha1_file(path)
    except Exception as e:
        print(f"Failed: {path} :: {e}")
        return None


# ==========================================================
# BUILD TARGET HASH SET
# ==========================================================

target_files = [
    p
    for p in Path(TARGET_DIR).rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]

print(f"Hashing {len(target_files):,} target images...")

target_hashes = set()

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    for h in tqdm(
        executor.map(hash_file_safe, target_files),
        total=len(target_files),
        desc="Target hashes",
    ):
        if h:
            target_hashes.add(h)

# ==========================================================
# CHECK SOURCE FILES
# ==========================================================

source_files = [
    p
    for p in Path(SOURCE_DIR).rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]

print(f"\nChecking {len(source_files):,} source images...")


def source_match(path):
    try:
        return sha1_file(path) in target_hashes
    except Exception as e:
        print(f"Failed: {path} :: {e}")
        return None


matches = 0
missing = 0

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    for result in tqdm(
        executor.map(source_match, source_files),
        total=len(source_files),
        desc="Source check",
    ):
        if result is True:
            matches += 1
        elif result is False:
            missing += 1

# ==========================================================
# RESULTS
# ==========================================================

total = matches + missing

pct = 100.0 * matches / total if total else 0.0

print("\n========================")
print(f"Source Images : {total:,}")
print(f"Matched       : {matches:,}")
print(f"Missing       : {missing:,}")
print(f"Match Rate    : {pct:.2f}%")
print("========================")