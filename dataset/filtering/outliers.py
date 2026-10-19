import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

#--------------------------------------------------------------
# Config
#--------------------------------------------------------------

# CSV from running test.py
EFFICIENTNET_CSV = "eval_outputs_efficientnet_b4_realonly.pth_serenity/age_predictions.csv"
OUTPUT_DIR = "dataset/generation/gen/stats_age/serenity"

OUTPUT_CSV_1STDDEV = False
OUTPUT_CSV_2STDDEV = False
OUTPUT_CSV_3STDDEV = False
OUTPUT_CSV_4STDDEV = True
OUTPUT_CSV_5STDDEV = True

COPY_OUTLIERS = True
COPY_STDDEV_LEVEL = 4

DELETE_OUTLIERS = True
DELETE_STDDEV_LEVEL = 4

#-------------------------------------------------------------------
# Stats Aggregation
#-------------------------------------------------------------------

df = pd.read_csv(EFFICIENTNET_CSV)
sample_stats = []
age_stats = []
total_samples = 0
total_within_1std = 0
total_within_2std = 0
total_within_3std = 0
total_within_4std = 0
total_within_5std = 0

for age, group in df.groupby("target_age"):
    mean_detected = group["detected_age"].mean()
    std_detected = group["detected_age"].std()
    dev_from_mean = (group["detected_age"] - mean_detected).abs()

    within_1std = (dev_from_mean <= std_detected)
    within_2std = (dev_from_mean <= 2 * std_detected)
    within_3std = (dev_from_mean <= 3 * std_detected)
    within_4std = (dev_from_mean <= 4 * std_detected)
    within_5std = (dev_from_mean <= 5 * std_detected)

    total_samples += len(group)
    total_within_1std += within_1std.sum()
    total_within_2std += within_2std.sum()
    total_within_3std += within_3std.sum()
    total_within_4std += within_4std.sum()
    total_within_5std += within_5std.sum()

    age_stats.append({
        "target_age": age,
        "within_1std": within_1std.mean(),
        "within_2std": within_2std.mean(),
        "within_3std": within_3std.mean(),
        "within_4std": within_4std.mean(),
        "within_5std": within_5std.mean(),
        "std": std_detected,
        "mean_detected": mean_detected,
        "count": len(group)
    })

    for idx, row in group.iterrows():
        stddevs_from_mean = 0 if std_detected == 0 else (
            abs(row["detected_age"] - mean_detected) / std_detected
        )
        sample_stats.append({
            "file_path": row["file_path"],
            "target_age": row["target_age"],
            "detected_age": row["detected_age"],
            "mean_detected": mean_detected,
            "std_detected": std_detected,
            "stddevs_from_mean": stddevs_from_mean,
        })

sample_stats = pd.DataFrame(sample_stats)
age_stats = pd.DataFrame(age_stats)

#---------------------------------------------------
# Print stats
#---------------------------------------------------
plt.figure(figsize=(12,6))

plt.plot(
    age_stats["target_age"],
    age_stats["within_1std"],
    label="Within ±1 std dev"
)  
plt.plot(
    age_stats["target_age"],
    age_stats["within_2std"],
    label="Within ±2 std dev"
)
plt.plot(
    age_stats["target_age"],
    age_stats["within_3std"],
    label="Within ±3 std dev"
)   
plt.plot(
    age_stats["target_age"],
    age_stats["within_4std"],
    label="Within ±4 std dev"
)
plt.plot(
    age_stats["target_age"],
    age_stats["within_5std"],
    label="Within ±5 std dev"
)

plt.xlabel("Target Age")
plt.ylabel("Fraction of Predictions")
plt.title("Prediction Consistency by Age")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(f"{OUTPUT_DIR}/prediction_consistency.png", dpi=300)
plt.close()

print("\n=== Overall Consistency ===")
print(
    f"Within ±1σ: "
    f"{total_within_1std:,}/{total_samples:,} "
    f"({total_within_1std / total_samples:.2%})"
)
print(
    f"Within ±2σ: "
    f"{total_within_2std:,}/{total_samples:,} "
    f"({total_within_2std / total_samples:.2%})"
)
print(
    f"Within ±3σ: "
    f"{total_within_3std:,}/{total_samples:,} "
    f"({total_within_3std / total_samples:.2%})"
)
print(
    f"Within ±4σ: "
    f"{total_within_4std:,}/{total_samples:,} "
    f"({total_within_4std / total_samples:.2%})"
)
print(
    f"Within ±5σ: "
    f"{total_within_5std:,}/{total_samples:,} "
    f"({total_within_5std / total_samples:.2%})"
)

#-----------------------------------------------------
# CSV
#-----------------------------------------------------
csv_configs = [
    (OUTPUT_CSV_1STDDEV, 1),
    (OUTPUT_CSV_2STDDEV, 2),
    (OUTPUT_CSV_3STDDEV, 3),
    (OUTPUT_CSV_4STDDEV, 4),
    (OUTPUT_CSV_5STDDEV, 5),
]

for enabled, level in csv_configs:
    if not enabled:
        continue

    outliers = sample_stats[
        sample_stats["stddevs_from_mean"] >= level
    ].sort_values(
        "stddevs_from_mean",
        ascending=False
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"outliers_{level}stddev.csv"
    )

    outliers.to_csv(csv_path, index=False)

    print(
        f"Saved {len(outliers):,} rows -> {csv_path}"
    )

#-----------------------------------------------------
# Copy files for manual checking
#-----------------------------------------------------
if COPY_OUTLIERS:
    copy_df = sample_stats[
        sample_stats["stddevs_from_mean"] >= COPY_STDDEV_LEVEL
    ]
    copy_root = os.path.join(
        OUTPUT_DIR,
        f"{COPY_STDDEV_LEVEL}stddev_outliers"
    )
    copied = 0

    for _, row in copy_df.iterrows():
        src = row["file_path"]
        if not os.path.exists(src):
            continue
        age = str(int(round(row["target_age"])))
        dst = os.path.join(
            copy_root,
            age,
            os.path.basename(src)
        )
        os.makedirs(
            os.path.dirname(dst),
            exist_ok=True
        )
        shutil.copy2(src, dst)
        copied += 1

    print(
        f"Copied {copied:,} files to {copy_root}"
    )

#-----------------------------------------------------
# Delete outliers from source dir
#-----------------------------------------------------
if DELETE_OUTLIERS:
    deletion_stats = age_stats[["target_age", "std"]].copy()
    deletion_stats["std_dev_threshold"] = (
        deletion_stats["std"] * DELETE_STDDEV_LEVEL
    )
    deletion_stats = deletion_stats.rename(columns={"target_age": "age"})
    deletion_stats[["age", "std_dev_threshold"]].to_csv(
        os.path.join(OUTPUT_DIR, "deletion_stats.csv"),
        index=False
    )

    delete_df = sample_stats[
        sample_stats["stddevs_from_mean"] >= DELETE_STDDEV_LEVEL
    ]

    deleted = 0
    missing = 0

    for _, row in delete_df.iterrows():
        path = row["file_path"]

        try:
            os.remove(path)
            deleted += 1
        except FileNotFoundError:
            missing += 1

    print(
        f"Deleted {deleted:,} files "
        f"(missing: {missing:,}) "
        f"at >= {DELETE_STDDEV_LEVEL}σ"
    )