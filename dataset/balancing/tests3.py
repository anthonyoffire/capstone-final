import os
import shutil
import pandas as pd
from tqdm import tqdm

CSV_PATH = "../images/real/uniform_split/age_stats/mivolo_non_detections.csv"
SOURCE_ROOT = "../images/real/uniform_split/all"
DEST_ROOT = "../images/real/uniform_split/all-minus-noface"

# Build set of files to exclude
df = pd.read_csv(CSV_PATH)

excluded = {
    (str(int(row["target_age"])), row["file"])
    for _, row in df.iterrows()
}

# Count files for tqdm
total_files = 0
for age_dir in os.listdir(SOURCE_ROOT):
    age_path = os.path.join(SOURCE_ROOT, age_dir)

    if os.path.isdir(age_path):
        total_files += len(os.listdir(age_path))

copied = 0
skipped = 0

with tqdm(total=total_files, desc="Copying") as pbar:
    for age_dir in os.listdir(SOURCE_ROOT):
        age_path = os.path.join(SOURCE_ROOT, age_dir)

        if not os.path.isdir(age_path):
            continue

        for filename in os.listdir(age_path):
            pbar.update(1)

            key = (age_dir, filename)

            if key in excluded:
                skipped += 1
                continue

            src = os.path.join(age_path, filename)
            dst = os.path.join(DEST_ROOT, age_dir, filename)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

            copied += 1

print(f"Copied: {copied}")
print(f"Skipped (no detections): {skipped}")