import os
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================================
# CONFIG
# ==========================================================

COUNT_ONLY = True
INPUT_DIR = (PROJECT_ROOT / "../images/fake/synth_serenity_85").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "../images/fake/ser_stats_age").resolve()
IMAGE_SIZE = 128
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEVICE = "cuda"
CUSTOM_MODEL_PATH = PROJECT_ROOT / "best_models" / "efficientnet_b4_realonly.pth"
MIN_AGE = 1
MAX_AGE = 100
AGE_RANGE = 100

if not COUNT_ONLY:
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm
    from types import SimpleNamespace

    sys.path.insert(0, str(PROJECT_ROOT))
    from predictor.model import build_model

    MIVOLO_DIR = Path.home() / "MiVOLO"
    sys.path.insert(0, str(MIVOLO_DIR))
    from mivolo.predictor import Predictor

    args = SimpleNamespace(
        detector_weights="dataset/generation/pretrained_models/yolov8x_person_face.pt",
        checkpoint="dataset/generation/pretrained_models/model_imdb_cross_person_4.22_99.46.pth.tar",
        with_persons=True,
        disable_faces=False,
        device="cuda:0",
        draw=False,
        num_workers=4
    )
    predictor = Predictor(args, verbose=False)

    def get_custom_transforms():
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def load_custom_model(model_p):
        save_model = torch.load(model_p, map_location=DEVICE)
        model = build_model("efficientnet_b4").to(DEVICE)
        model.load_state_dict(save_model["model_state_dict"])
        model.eval()
        return model

    custom_model = load_custom_model(CUSTOM_MODEL_PATH)
    custom_transforms = get_custom_transforms()

# ==========================================================
# AGE ESTIMATORS
# ==========================================================

def predict_age_mivolo(img_path):
    img = cv2.imread(str(Path(img_path).resolve()))
    if img is None:
        return None
    detected_objects, _ = predictor.recognize(img)
    if len(detected_objects.ages) == 0:
        print(f"No people identified: {img_path}")
        return None
    return float(detected_objects.ages[0])

def predict_age_custom(img_path):
    img = Image.open(str(Path(img_path).resolve())).convert("RGB")
    tensor = custom_transforms(img)
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = custom_model(tensor)
        age = pred.item() * AGE_RANGE + MIN_AGE
        return float(age)

ESTIMATORS = {
    "mivolo": predict_age_mivolo,
    "custom": predict_age_custom,
}

def predict_age(img_path, estimator):
    if estimator == "mivolo":
        return predict_age_mivolo(img_path)
    return predict_age_custom(img_path)

def process_image(img_path, target_age, model_name, estimator_name, predict_age):
    try:
        detected_age = predict_age(str(img_path))
    except Exception as e:
        return ("error", f"{img_path.name}: {e}")

    if detected_age is None:
        return ("none", {
            "target_age": target_age,
            "file": img_path.name,
        })

    return ("record", {
        "estimator": estimator_name,
        "model": model_name,
        "target_age": target_age,
        "detected_age": float(detected_age),
        "error": float(detected_age - target_age),
        "file": img_path.name,
    })


def get_model_age_dirs(input_root):
    def get_age_dirs(parent):
        age_dirs = []
        with os.scandir(parent) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    age_val = float(entry.name)
                except ValueError:
                    continue
                age_dirs.append((age_val, Path(entry.path)))
        return sorted(age_dirs)

    root_age_dirs = get_age_dirs(input_root)
    if root_age_dirs:
        return [(input_root.name, root_age_dirs)]

    model_age_dirs = []
    with os.scandir(input_root) as entries:
        model_dirs = sorted(
            (Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False)),
            key=lambda model_dir: model_dir.name,
        )
    for model_dir in model_dirs:
        model_age_dirs.append((model_dir.name, get_age_dirs(model_dir)))
    return model_age_dirs


def count_images(age_dir):
    with os.scandir(age_dir) as entries:
        return sum(
            1
            for entry in entries
            if entry.is_file(follow_symlinks=False)
            and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTS
        )


def run_count_only(input_root):
    model_age_dirs = get_model_age_dirs(input_root)
    ages = sorted({
        target_age
        for _, age_dirs in model_age_dirs
        for target_age, _ in age_dirs
    })
    if not ages:
        raise ValueError(f"No numeric age folders found in: {input_root}")

    counts_by_model = {}
    for model_name, age_dirs in model_age_dirs:
        counts_by_age = {age: 0 for age in ages}
        for target_age, age_dir in age_dirs:
            counts_by_age[target_age] = count_images(age_dir)
        counts_by_model[model_name] = counts_by_age

    plt.figure(figsize=(12, 8))
    for model_name, counts_by_age in counts_by_model.items():
        plt.plot(
            ages,
            [counts_by_age[age] for age in ages],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Target Age")
    plt.ylabel("Image Count")
    plt.title("Images per Target Age Bucket")
    tick_step = max(1, (len(ages) + 14) // 15)
    age_ticks = ages[::tick_step]
    if age_ticks[-1] != ages[-1]:
        age_ticks.append(ages[-1])
    plt.xticks(age_ticks)
    plt.grid(True, alpha=0.3)
    if len(counts_by_model) > 1:
        plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "image_count_by_age.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    total_images = sum(
        count
        for counts_by_age in counts_by_model.values()
        for count in counts_by_age.values()
    )
    print(f"Counted {total_images} images across {len(ages)} age buckets.")
    print(f"Saved: {output_path}")


# ==========================================================
# MAIN
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

input_root = Path(INPUT_DIR)
if not input_root.is_dir():
    raise FileNotFoundError(f"Input image directory does not exist: {input_root}")

if COUNT_ONLY:
    run_count_only(input_root)
    raise SystemExit(0)

for estimator_name, predict_age in ESTIMATORS.items():
    if estimator_name == "mivolo":
        continue

    records = []
    model_age_dirs = get_model_age_dirs(input_root)

    for model_name, age_dirs in model_age_dirs:
        print(f"\n=== {estimator_name} === {model_name} ===")
        age_dirs.sort()

        jobs = []
        for target_age, age_dir in age_dirs:
            for img_path in sorted(age_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                jobs.append(
                    (
                        img_path,
                        target_age,
                        model_name,
                        estimator_name,
                        predict_age,
                    )
                )

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_image, *job)
                for job in jobs
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{estimator_name} - {model_name}",
                unit="img",
                leave=True,
            ):
                kind, data = future.result()
                if kind == "record":
                    records.append(data)
                elif kind == "error":
                    print("oops:", data)
    # ==========================================================
    # DATAFRAME
    # ==========================================================

    if not records:
        print("No results, skipping stats...")
        continue
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, f"{estimator_name}_age_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ==========================================================
    # PER-MODEL STATS
    # ==========================================================

    summary_rows = []

    for model_name in sorted(df["model"].unique()):
        model_df = df[df["model"] == model_name]
        for target_age in sorted(model_df["target_age"].unique()):
            subset = model_df[
                model_df["target_age"] == target_age
            ]
            ages = subset["detected_age"].to_numpy()
            summary_rows.append({
                "model": model_name,
                "target_age": target_age,
                "count": len(ages),
                "mean": np.mean(ages),
                "median": np.median(ages),
                "std": np.std(ages),
                "min": np.min(ages),
                "max": np.max(ages),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        OUTPUT_DIR,
        f"{estimator_name}_age_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # ==========================================================
    # DETECTED AGE DISTRIBUTION (1-YEAR BINS)
    # ==========================================================

    plt.figure(figsize=(18, 8))

    models = sorted(df["model"].unique())
    min_age = int(df["detected_age"].min())
    max_age = int(df["detected_age"].max())
    ages = np.arange(min_age, max_age + 1)
    counts_by_model = {}
    for model_name in models:
        model_df = df[df["model"] == model_name]
        detected = (
            model_df["detected_age"]
            .round()
            .astype(int)
        )
        counts = [
            (detected == age).sum()
            for age in ages
        ]
        counts_by_model[model_name] = counts

    for model_name in models:
        plt.plot(
            ages,
            counts_by_model[model_name],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xticks(ages, rotation=90)

    plt.xlabel("Detected Age")
    plt.ylabel("Image Count")
    plt.title("Detected Age Distribution")

    plt.grid(axis="y", alpha=0.3)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{estimator_name}_detected_age_distribution.png"
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # ==========================================================
    # MEAN AGE VS TARGET
    # ==========================================================

    plt.figure(figsize=(12, 8))

    all_ages = sorted(summary_df["target_age"].unique())

    # Ideal line
    plt.plot(
        all_ages,
        all_ages,
        "k--",
        linewidth=2,
        label="Ideal (y=x)",
    )

    for model_name in sorted(summary_df["model"].unique()):

        model_stats = (
            summary_df[summary_df["model"] == model_name]
            .sort_values("target_age")
        )

        plt.plot(
            model_stats["target_age"],
            model_stats["mean"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Target Age")
    plt.ylabel("Mean Detected Age")
    plt.title("Target Age vs Mean Detected Age")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{estimator_name}_age_mean_comparison.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================================
    # MEDIAN AGE VS TARGET
    # ==========================================================

    plt.figure(figsize=(12, 8))

    all_ages = sorted(summary_df["target_age"].unique())

    # Ideal line
    plt.plot(
        all_ages,
        all_ages,
        "k--",
        linewidth=2,
        label="Ideal (y=x)",
    )

    for model_name in sorted(summary_df["model"].unique()):

        model_stats = (
            summary_df[summary_df["model"] == model_name]
            .sort_values("target_age")
        )

        plt.plot(
            model_stats["target_age"],
            model_stats["median"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Target Age")
    plt.ylabel("Median Detected Age")
    plt.title("Target Age vs Median Detected Age")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{estimator_name}_age_median_comparison.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()
    # ==========================================================
    # AGE STD DEV COMPARISON
    # ==========================================================

    plt.figure(figsize=(12, 8))

    for model_name in sorted(summary_df["model"].unique()):

        model_stats = (
            summary_df[summary_df["model"] == model_name]
            .sort_values("target_age")
        )

        plt.plot(
            model_stats["target_age"],
            model_stats["std"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Target Age")
    plt.ylabel("Detected Age Std Dev")
    plt.title("Age Variability by Target Age")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{estimator_name}_age_stddev_comparison.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()