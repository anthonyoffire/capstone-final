import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
import sys
from pathlib import Path
import cv2
from PIL import Image
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_DIR = PROJECT_ROOT / "predictor"
sys.path.insert(0, str(PROJECT_ROOT))
from predictor.model import build_model

MIVOLO_DIR = Path.home() / "MiVOLO"
sys.path.insert(0, str(MIVOLO_DIR))

from mivolo.predictor import Predictor

# ==========================================================
# CONFIG
# ==========================================================

INPUT_DIR = "dataset/generation/gen/images"
OUTPUT_DIR = "dataset/generation/gen/stats_age"
IMAGE_SIZE = 128
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEVICE = "cuda"
CUSTOM_MODEL_PATH = PROJECT_ROOT / "best_models" / "efficientnet_b4_realonly.pth"
MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = 65
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
    result = model.load_state_dict(
        save_model["model_state_dict"]
    )
    model.eval()
    return model
custom_model = load_custom_model(
    CUSTOM_MODEL_PATH
)
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
# ==========================================================
# MAIN
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for estimator_name, predict_age in ESTIMATORS.items():
    if estimator_name == "mivolo":
        continue

    input_root = Path(INPUT_DIR)
    records = []

    for model_dir in sorted(input_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        print(f"\n=== {estimator_name} === {model_name} ===")

        age_dirs = []
        for age_dir in model_dir.iterdir():
            if not age_dir.is_dir():
                continue
            try:
                age_val = float(age_dir.name)
            except ValueError:
                continue
            age_dirs.append((age_val, age_dir))
        age_dirs.sort()

        jobs = []
        for target_age, age_dir in age_dirs:
            for img_path in sorted(age_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
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

    x = np.arange(len(ages))
    bar_width = 0.8 / len(models)

    for i, model_name in enumerate(models):
        offsets = x + (i - (len(models) - 1) / 2) * bar_width
        plt.bar(
            offsets,
            counts_by_model[model_name],
            width=bar_width,
            label=model_name,
        )

    plt.xticks(
        x,
        ages,
        rotation=90,
    )

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