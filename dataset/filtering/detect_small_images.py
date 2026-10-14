import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ----------------------------------------
# Config
# ----------------------------------------

SRC_DIR = "../images/real/uniform_split/all-minus-noface"
MIN_SIZE = 60

WRITE_STAT_CSV = True
WRITE_OUTPUT_CSV = False
COPY_SMALL_IMAGES = False
DELETE_SMALL_IMAGES = True
OUTPUT_DIR = "../images/real/uniform_split/small_image_scan"
COPY_DIR = f"{OUTPUT_DIR}/images_under_{MIN_SIZE}"
STAT_CSV = "dataset/filtering/image_sizes.csv"      # cache of all image sizes
OUTPUT_CSV = "small_images.csv"   # current threshold results
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ----------------------------------------
# Data Aggregation
# ----------------------------------------
if not os.path.exists(STAT_CSV):
    print("Scanning images...")

    rows = []
    image_paths = []

    for root, _, files in os.walk(SRC_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, file))
    for path in tqdm(image_paths, desc="Scanning"):
        try:
            with Image.open(path) as img:
                width, height = img.size
            rows.append({
                "file_path": path,
                "size": min(width, height)
            })
        except Exception as e:
            print(f"Error reading {path}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(STAT_CSV, index=False)
    print(f"Saved: {STAT_CSV}")

else:
    print(f"Loading existing stats: {STAT_CSV}")
    df = pd.read_csv(STAT_CSV)
small_df = df[df["size"] < MIN_SIZE]

print(f"Pics < {MIN_SIZE}: {len(small_df)}")
# ----------------------------------------
# CSV
# ----------------------------------------

if WRITE_OUTPUT_CSV:
    csv_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_CSV
    )
    small_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

# ----------------------------------------
# Copy
# ----------------------------------------

if COPY_SMALL_IMAGES:
    os.makedirs(COPY_DIR, exist_ok=True)
    copied = 0
    for _, row in tqdm(
        small_df.iterrows(),
        total=len(small_df),
        desc="Copying"
    ):
        src = row["file_path"]
        rel_path = os.path.relpath(src, SRC_DIR)
        dst = os.path.join(
            COPY_DIR,
            rel_path
        )
        os.makedirs(
            os.path.dirname(dst),
            exist_ok=True
        )
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied: {copied:,}")

# ----------------------------------------
# Delete
# ----------------------------------------

if DELETE_SMALL_IMAGES:
    deleted = 0
    missing = 0
    for path in tqdm(
        small_df["file_path"],
        desc="Deleting"
    ):
        try:
            os.remove(path)
            deleted += 1
        except FileNotFoundError:
            missing += 1
    print(
        f"Deleted: {deleted:,} "
        f"(missing: {missing:,})"
    )