import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

import os
from collections import defaultdict
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import pandas as pd
from dataset import AgeDataset
from model import build_model
from image_shuffler import PIL_Shuffler, validate_grid_size
import matplotlib.pyplot as plt
from pathlib import Path
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE

IMAGE_SIZE = 128
BATCH_SIZE = 32
GRID_SIZE = 1
MODEL_NAME = "efficientnet_b4"

def get_evaluation_transforms():
    return transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    PIL_Shuffler(grid_size=GRID_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model(model_p):
    save_model = torch.load(model_p, map_location=DEVICE)
    model = build_model(MODEL_NAME).to(DEVICE)
    model.load_state_dict(save_model["model_state_dict"])
    model.eval()
    return model

def eval(csv_path, model_path, age_thresholds=(18,), output_dir="./eval_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    trans = get_evaluation_transforms()
    evaluation_dataset = AgeDataset(csv_path, transform=trans)

    loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = load_model(model_path)

    errors = []
    errors_by_age = defaultdict(list)

    all_true_years = []
    all_pred_years = []
    all_file_paths = []

    with torch.no_grad():
        for images, ages, file_paths in tqdm(loader, desc="testing"):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

            preds = model(images)

            pred_years = preds * AGE_RANGE + MIN_AGE
            true_years = ages * AGE_RANGE + MIN_AGE

            batch_errors = (pred_years - true_years).cpu().numpy().flatten()
            batch_true_years = true_years.cpu().numpy().flatten()
            batch_pred_years = pred_years.cpu().numpy().flatten()

            errors.extend(batch_errors)
            all_true_years.extend(batch_true_years.tolist())
            all_pred_years.extend(batch_pred_years.tolist())
            all_file_paths.extend(file_paths)

            for true_age, pred_age, err in zip(batch_true_years, batch_pred_years, batch_errors):
                age_bucket = int(round(true_age))
                errors_by_age[age_bucket].append(err)
    errors = np.array(errors)
    all_true_years = np.array(all_true_years)
    all_pred_years = np.array(all_pred_years)

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    medae = np.median(np.abs(errors))
    bias = np.mean(errors)
    std_err = np.std(errors)
    max_err = np.max(np.abs(errors))

    print("\n=== Overall Regression Metrics ===")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Median AE: {medae:.2f}")
    print(f"Bias: {bias:.2f}")
    print(f"Std Error: {std_err:.2f}")
    print(f"Max Error: {max_err:.2f}")
    print(f"Samples: {len(errors)}")

    per_age_reg_path = os.path.join(output_dir, "per_age_regression.csv")
    with open(per_age_reg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["age", "count", "mae", "bias", "rmse"])

        for age in sorted(errors_by_age.keys()):
            age_errors = np.array(errors_by_age[age])
            writer.writerow([
                age,
                len(age_errors),
                np.mean(np.abs(age_errors)),
                np.mean(age_errors),
                np.sqrt(np.mean(age_errors ** 2))
            ])

    print(f"\nSaved: {per_age_reg_path}")

    summary_rows = []
    per_age_cls_rows = []

    for threshold in age_thresholds:
        true_cls = all_true_years >= threshold
        pred_cls = all_pred_years >= threshold

        tp = int(np.sum(pred_cls & true_cls))
        tn = int(np.sum((~pred_cls) & (~true_cls)))
        fp = int(np.sum(pred_cls & (~true_cls)))
        fn = int(np.sum((~pred_cls) & true_cls))

        total = len(all_true_years)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n=== Classification @ {threshold} ===")
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        summary_rows.append([
            threshold, accuracy, precision, recall, specificity, f1, tp, tn, fp, fn
        ])

        # per-age classification
        for age in sorted(set(int(round(x)) for x in all_true_years)):
            mask = np.round(all_true_years).astype(int) == age
            if np.sum(mask) == 0:
                continue

            age_true = all_true_years[mask]
            age_pred = all_pred_years[mask]

            age_acc = np.mean((age_true >= threshold) == (age_pred >= threshold))

            per_age_cls_rows.append([
                threshold, age, np.sum(mask), age_acc
            ])

    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold", "accuracy", "precision", "recall",
            "specificity", "f1", "tp", "tn", "fp", "fn"
        ])
        writer.writerows(summary_rows)

    print(f"Saved: {summary_path}")

    per_age_cls_path = os.path.join(output_dir, "per_age_classification.csv")
    with open(per_age_cls_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "age", "count", "accuracy"])
        writer.writerows(per_age_cls_rows)

    print(f"Saved: {per_age_cls_path}")

    # --------------------------------------------------
    # Save all predictions
    # --------------------------------------------------

    predictions_df = pd.DataFrame({
        "target_age": all_true_years,
        "detected_age": all_pred_years,
        "error": all_pred_years - all_true_years,
        "file_path": all_file_paths,
    })

    predictions_path = os.path.join(output_dir, "age_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    print(f"Saved: {predictions_path}")

    # --------------------------------------------------
    # Mean predicted age vs target age
    # --------------------------------------------------

    mean_predictions = (
        predictions_df
        .assign(target_age_round=np.round(predictions_df["target_age"]).astype(int))
        .groupby("target_age_round")["detected_age"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        mean_predictions["target_age_round"],
        mean_predictions["detected_age"],
        label="Mean Prediction"
    )

    # ideal line
    min_age = mean_predictions["target_age_round"].min()
    max_age = mean_predictions["target_age_round"].max()

    plt.plot(
        [min_age, max_age],
        [min_age, max_age],
        "--",
        label="Perfect Prediction"
    )

    plt.xlabel("Target Age")
    plt.ylabel("Mean Detected Age")
    plt.title("Target Age vs Mean Detected Age")
    plt.legend()
    plt.grid(True)

    mean_plot_path = os.path.join(
        output_dir,
        "target_vs_mean_detected_age.png"
    )

    plt.savefig(mean_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {mean_plot_path}")

    # --------------------------------------------------
    # Detected age distribution
    # --------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.hist(
        all_pred_years,
        bins=50,
    )

    plt.xlabel("Detected Age")
    plt.ylabel("Count")
    plt.title("Detected Age Distribution")
    plt.grid(True)

    dist_plot_path = os.path.join(
        output_dir,
        "detected_age_distribution.png"
    )

    plt.savefig(dist_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {dist_plot_path}")

def singlepred(filepath, modelpath):
    model = load_model(modelpath)
    pred_img = Image.open(filepath).convert('RGB')
    pred_img = get_evaluation_transforms()(pred_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(pred_img)
    print(pred)
    print(pred.shape)
    print(pred.item())
    predicted_age = pred.item() * AGE_RANGE + MIN_AGE
    print(f"Predicted Age: {round(predicted_age)}")


if __name__ == '__main__':
    csv_path = "./synthetic_csv/serenity.csv"
    model_path = "./best_models/efficientnet_b4_realonly.pth"
    parser = argparse.ArgumentParser(description="Run testing on a shuffled image set.")
    parser.add_argument(
        "-g", "--grid_size",
        type=validate_grid_size,
        default=1, # No scramble
        help="Grid size (e.g., 4 creates a 4x4 grid). Must be a power of 2 <= 128, > 0"
    )
    args = parser.parse_args()
    GRID_SIZE = args.grid_size
    checkpoint_title = Path(model_path).name.removesuffix(".csv")
    test_name = Path(csv_path).name.removesuffix(".csv")
    eval(csv_path, model_path, age_thresholds=(13, 15, 18, 21), output_dir=f"./eval_outputs_{checkpoint_title}_{test_name}")
